import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand, VerticalPerturbation, LateralPerturbation, DirectionalPerturbation

import config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if config.ROBOT == "h1" or config.ROBOT == "g1":
    perturb_body_id = mj_model.body("torso_link").id
else:
    perturb_body_id = mj_model.body("base_link").id

# Collect key callbacks from enabled perturbation systems
key_callbacks = []

if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    band_attached_link = perturb_body_id
    key_callbacks.append(elastic_band.MujuocoKeyCallback)

if config.ENABLE_VERTICAL_PERTURBATION:
    vertical_perturbation = VerticalPerturbation()
    key_callbacks.append(vertical_perturbation.MujocoKeyCallback)

if config.ENABLE_LATERAL_PERTURBATION:
    lateral_perturbation = LateralPerturbation()
    key_callbacks.append(lateral_perturbation.MujocoKeyCallback)

if config.ENABLE_DIRECTIONAL_PERTURBATION:
    directional_perturbation = DirectionalPerturbation()
    key_callbacks.append(directional_perturbation.MujocoKeyCallback)

if key_callbacks:
    def combined_key_callback(key):
        for cb in key_callbacks:
            cb(key)
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=combined_key_callback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

# Force visualization state (shared between sim and viewer threads)
force_vis_lock = threading.Lock()
force_vis_origin = np.zeros(3)  # body position in world frame
force_vis_vector = np.zeros(3)  # force vector in world frame


time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        mj_data.xfrc_applied[perturb_body_id, :] = 0  # clear previous forces

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] += elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )

        if config.ENABLE_VERTICAL_PERTURBATION:
            mj_data.xfrc_applied[perturb_body_id, :3] += vertical_perturbation.Advance(
                mj_model.opt.timestep
            )

        if config.ENABLE_LATERAL_PERTURBATION:
            mj_data.xfrc_applied[perturb_body_id, :3] += lateral_perturbation.Advance(
                mj_model.opt.timestep
            )

        if config.ENABLE_DIRECTIONAL_PERTURBATION:
            mj_data.xfrc_applied[perturb_body_id, :3] += directional_perturbation.Advance(
                mj_model.opt.timestep
            )

        # Capture force data for visualization
        with force_vis_lock:
            force_vis_origin[:] = mj_data.xpos[perturb_body_id]
            force_vis_vector[:] = mj_data.xfrc_applied[perturb_body_id, :3]

        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    ARROW_SCALE = 0.01  # meters per Newton
    ARROW_WIDTH = 0.015
    ARROW_RGBA = np.array([1.0, 0.2, 0.2, 0.8], dtype=np.float32)  # red

    COM_RADIUS = 0.025
    COM_RGBA = np.array([1.0, 0.2, 0.2, 0.9], dtype=np.float32)  # red — actual CoM
    COMPLIANT_COM_RGBA = np.array([0.2, 1.0, 0.2, 0.9], dtype=np.float32)  # green — compliant target
    COMPLIANCE_STIFFNESS = 500.0  # base stiffness (N/m), matches ComplianceManagerCfg
    robot_root_id = perturb_body_id  # base_link for Go2

    while viewer.is_running():
        locker.acquire()

        # Read force state
        with force_vis_lock:
            origin = force_vis_origin.copy()
            force = force_vis_vector.copy()

        force_mag = np.linalg.norm(force)
        com_pos = mj_data.subtree_com[robot_root_id].copy()

        # MSD steady-state displacement from current position: delta = F / k
        compliant_com = com_pos + force / COMPLIANCE_STIFFNESS

        with viewer.lock():
            viewer.sync()
            viewer.user_scn.ngeom = 0  # clear previous frame's geoms

            # Draw actual CoM (red)
            viewer.user_scn.ngeom += 1
            com_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
            com_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            mujoco.mjv_initGeom(
                geom=com_geom,
                type=mujoco.mjtGeom.mjGEOM_SPHERE.value,
                size=np.array([COM_RADIUS, 0, 0]),
                pos=com_pos.astype(np.float64),
                mat=np.eye(3).flatten(),
                rgba=COM_RGBA,
            )
            # Draw compliant target CoM (green) — MSD equilibrium under applied force
            viewer.user_scn.ngeom += 1
            comp_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
            comp_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            mujoco.mjv_initGeom(
                geom=comp_geom,
                type=mujoco.mjtGeom.mjGEOM_SPHERE.value,
                size=np.array([COM_RADIUS, 0, 0]),
                pos=compliant_com.astype(np.float64),
                mat=np.eye(3).flatten(),
                rgba=COMPLIANT_COM_RGBA,
            )

            if force_mag > 0.1:
                end = origin + force * ARROW_SCALE

                # Draw force arrow
                viewer.user_scn.ngeom += 1
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
                geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                mujoco.mjv_initGeom(
                    geom=geom,
                    type=mujoco.mjtGeom.mjGEOM_ARROW.value,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.zeros(9),
                    rgba=ARROW_RGBA,
                )
                mujoco.mjv_connector(
                    geom=geom,
                    type=mujoco.mjtGeom.mjGEOM_ARROW.value,
                    width=ARROW_WIDTH,
                    from_=origin.astype(np.float64),
                    to=end.astype(np.float64),
                )

            # Force info text floating above the robot
            fx, fy, fz = force
            hud_pos = origin.copy()
            hud_pos[2] += 0.35  # above the robot

            viewer.user_scn.ngeom += 1
            hud_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
            hud_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            mujoco.mjv_initGeom(
                geom=hud_geom,
                type=mujoco.mjtGeom.mjGEOM_SPHERE.value,
                size=np.array([0.001, 0, 0]),
                pos=hud_pos.astype(np.float64),
                mat=np.eye(3).flatten(),
                rgba=np.array([0, 0, 0, 0], dtype=np.float32),
            )
            hud_geom.label = (
                f"|F|={force_mag:.1f}  "
                f"Fx={fx:+.1f}  "
                f"Fy={fy:+.1f}  "
                f"Fz={fz:+.1f}"
            )

        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
