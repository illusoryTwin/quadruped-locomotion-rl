import json
import socket
import time
import os
import csv
from datetime import datetime
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand, VerticalPerturbation, LateralPerturbation, DirectionalPerturbation

import config
import info_panel


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if config.ROBOT == "h1" or config.ROBOT == "g1":
    perturb_body_id = mj_model.body("torso_link").id
else:
    perturb_body_id = mj_model.body("base_link").id

class StiffnessKeyController:
    """[ / ] keys to decrease / increase stiffness_commands in the running policy."""

    GLFW_KEY_LEFT_BRACKET  = 91
    GLFW_KEY_RIGHT_BRACKET = 93

    def __init__(self, step: float = 100.0, min_val: float = 0.0, max_val: float = 2000.0):
        self._step = step
        self._min = min_val
        self._max = max_val
        self._port = int(os.environ.get("STIFFNESS_PORT", "7777"))
        print(f"[StiffnessKeys] [ = −{step:.0f}   ] = +{step:.0f}   (port {self._port})")

    def _current(self) -> float:
        try:
            with open("/tmp/quadruped_stiffness") as f:
                return float(f.read().strip())
        except Exception:
            return 1000.0

    def _send(self, value: float):
        payload = json.dumps({"stiffness_commands": value}) + "\n"
        try:
            with socket.create_connection(("127.0.0.1", self._port), timeout=1.0) as sock:
                sock.sendall(payload.encode())
                sock.makefile().readline()
            print(f"[StiffnessKeys] stiffness_commands -> {value:.1f}")
        except Exception as e:
            print(f"[StiffnessKeys] send error: {e}")

    def MujocoKeyCallback(self, key):
        if key == self.GLFW_KEY_LEFT_BRACKET:
            self._send(max(self._min, self._current() - self._step))
        elif key == self.GLFW_KEY_RIGHT_BRACKET:
            self._send(min(self._max, self._current() + self._step))


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

if config.ENABLE_STIFFNESS_KEYS:
    stiffness_keys = StiffnessKeyController(step=config.STIFFNESS_KEY_STEP)
    key_callbacks.append(stiffness_keys.MujocoKeyCallback)

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

# CSV logging: sim time, base z, applied Fz, and commanded z target
LOG_DIR = os.environ.get("LOG_DIR", "/workspace/logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"mujoco_base_z_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
CMD_Z_TARGET = float(os.environ.get("CMD_Z_TARGET", "0.3"))
_log_file = open(LOG_PATH, "w", newline="", buffering=1)
_log_writer = csv.writer(_log_file)
_log_writer.writerow(["sim_time", "base_z", "force_fz", "commanded_z"])
print(f"[INFO] Logging MuJoCo base-z CSV to: {LOG_PATH}")
print(f"[INFO] Commanded z target for logging: {CMD_Z_TARGET:.4f}")


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

        # Log ground-truth base height and currently applied vertical force
        base_z = float(mj_data.xpos[perturb_body_id, 2])
        force_fz = float(mj_data.xfrc_applied[perturb_body_id, 2])
        _log_writer.writerow([f"{mj_data.time:.6f}", f"{base_z:.6f}", f"{force_fz:.6f}", f"{CMD_Z_TARGET:.6f}"])

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
        try:
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
                # viewer.user_scn.ngeom += 1
                # comp_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
                # comp_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                # mujoco.mjv_initGeom(
                    # geom=comp_geom,
                    # type=mujoco.mjtGeom.mjGEOM_SPHERE.value,
                    # size=np.array([COM_RADIUS, 0, 0]),
                    # pos=compliant_com.astype(np.float64),
                    # mat=np.eye(3).flatten(),
                    # rgba=COMPLIANT_COM_RGBA,
                # )
#
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

                # --- Forces label (upper) ---
                forces_pos = origin.copy()
                forces_pos[2] += 0.42

                viewer.user_scn.ngeom += 1
                forces_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
                forces_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                mujoco.mjv_initGeom(
                    geom=forces_geom,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE.value,
                    size=np.array([0.001, 0, 0]),
                    pos=forces_pos.astype(np.float64),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 0, 0, 0], dtype=np.float32),
                )
                fx, fy, fz = force
                forces_geom.label = f"|F|{force_mag:5.1f}N  Fx{fx:+.1f}  Fy{fy:+.1f}  Fz{fz:+.1f}"

                # Read stiffness / kd for both the in-scene label and the panel
                try:
                    with open("/tmp/quadruped_stiffness") as _f:
                        _stiffness = float(_f.read().strip())
                except Exception:
                    try:
                        _stiffness = float(os.environ.get("MUJOCO_HUD_KP", "nan"))
                    except Exception:
                        _stiffness = float("nan")
                try:
                    _kd = float(os.environ.get("MUJOCO_HUD_KD", "nan"))
                except Exception:
                    _kd = float("nan")

                # Update info panel
                info_panel.panel_state['force_mag'] = float(force_mag)
                info_panel.panel_state['fx']        = float(fx)
                info_panel.panel_state['fy']        = float(fy)
                info_panel.panel_state['fz']        = float(fz)
                info_panel.panel_state['stiffness'] = _stiffness
                info_panel.panel_state['kd']        = _kd

                # --- Stiffness label (lower, in-scene) ---
                stiff_pos = origin.copy()
                stiff_pos[2] += 0.35

                viewer.user_scn.ngeom += 1
                stiff_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
                stiff_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                mujoco.mjv_initGeom(
                    geom=stiff_geom,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE.value,
                    size=np.array([0.001, 0, 0]),
                    pos=stiff_pos.astype(np.float64),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 0, 0, 0], dtype=np.float32),
                )
                stiff_ok  = (_stiffness == _stiffness)
                kd_ok     = (_kd == _kd)
                BAR_LEN   = 11
                STIFF_ZERO = 400.0
                STIFF_STEP = 100.0
                filled    = max(0, min(BAR_LEN, int((_stiffness - STIFF_ZERO) / STIFF_STEP))) if stiff_ok else 0
                bar       = "#" * filled + "." * (BAR_LEN - filled)
                stiff_str = f"{_stiffness:.0f}" if stiff_ok else "n/a"
                kd_str    = f"{_kd:.3f}"        if kd_ok    else "n/a"
                step_str  = f"+/-{config.STIFFNESS_KEY_STEP:.0f}" if config.ENABLE_STIFFNESS_KEYS else ""
                stiff_line1 = f"kp {stiff_str}  [{bar}]  kd {kd_str}"
                stiff_line2 = f"            [  /  ] {step_str}" if config.ENABLE_STIFFNESS_KEYS else ""
                stiff_geom.label = "\n".join(x for x in [stiff_line1, stiff_line2] if x)
        finally:
            locker.release()
        time.sleep(config.VIEWER_DT)


def InfoPanelThread():
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,920")
    pygame.init()

    W, H = 500, 240
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Robot Status")

    # Colors
    BG        = (15, 18, 28)
    DIV       = (35, 40, 58)
    SEC_COL   = (110, 120, 150)
    MAG_COL   = (170, 200, 255)   # |F| magnitude
    FX_COL    = (255, 130, 110)   # Fx  warm red
    FY_COL    = (110, 230, 145)   # Fy  green
    FZ_COL    = (110, 175, 255)   # Fz  blue
    KP_COL    = (90, 235, 135)
    KD_COL    = (170, 175, 200)
    BAR_BG    = (38, 44, 62)
    BAR_FILL  = (75, 195, 115)
    HINT_COL  = (85, 92, 118)

    font_sec   = pygame.font.SysFont("dejavusans", 12, bold=True)
    font_val   = pygame.font.SysFont("dejavusans", 14)
    font_large = pygame.font.SysFont("dejavusans", 24, bold=True)
    font_hint  = pygame.font.SysFont("dejavusans", 12)

    BAR_X, BAR_Y, BAR_W, BAR_H = 168, 152, 190, 16
    STIFF_ZERO, STIFF_FULL = 400.0, 1500.0

    clock = pygame.time.Clock()

    while viewer.is_running():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        s         = _panel_state
        stiffness = s['stiffness']
        kd        = s['kd']
        force_mag = s['force_mag']
        fx, fy, fz = s['fx'], s['fy'], s['fz']
        stiff_ok  = (stiffness == stiffness)
        kd_ok     = (kd == kd)

        screen.fill(BG)

        # --- Forces section ---
        screen.blit(font_sec.render("FORCES", True, SEC_COL), (15, 10))
        screen.blit(font_large.render(f"|F|  {force_mag:.1f} N", True, MAG_COL), (15, 26))

        # Fx / Fy / Fz — each with its own accent color
        col_x = 15
        for label, val, col in (("Fx", fx, FX_COL), ("Fy", fy, FY_COL), ("Fz", fz, FZ_COL)):
            surf_lbl = font_sec.render(label, True, SEC_COL)
            surf_val = font_val.render(f"{val:+.1f}", True, col)
            screen.blit(surf_lbl, (col_x, 64))
            screen.blit(surf_val, (col_x, 78))
            col_x += 155

        pygame.draw.line(screen, DIV, (15, 108), (W - 15, 108), 1)

        # --- Stiffness section ---
        screen.blit(font_sec.render("STIFFNESS", True, SEC_COL), (15, 118))

        kp_str = f"{stiffness:.0f}" if stiff_ok else "n/a"
        screen.blit(font_large.render(f"kp  {kp_str}", True, KP_COL), (15, 132))

        # Progress bar
        pygame.draw.rect(screen, BAR_BG, (BAR_X, BAR_Y, BAR_W, BAR_H), border_radius=4)
        if stiff_ok:
            fill_w = int(max(0, min(BAR_W, (stiffness - STIFF_ZERO) / (STIFF_FULL - STIFF_ZERO) * BAR_W)))
            if fill_w > 0:
                pygame.draw.rect(screen, BAR_FILL, (BAR_X, BAR_Y, fill_w, BAR_H), border_radius=4)

        kd_str = f"kd  {kd:.3f}" if kd_ok else "kd  n/a"
        screen.blit(font_val.render(kd_str, True, KD_COL), (BAR_X + BAR_W + 12, BAR_Y))

        # Key hint
        if config.ENABLE_STIFFNESS_KEYS:
            hint = f"[ = less           ] = more           ±{config.STIFFNESS_KEY_STEP:.0f}"
            screen.blit(font_hint.render(hint, True, HINT_COL), (15, 200))

        pygame.display.flip()
        clock.tick(50)

    pygame.quit()


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread    = Thread(target=SimulationThread)
    panel_thread  = Thread(target=info_panel.run, args=(viewer,), daemon=True)

    viewer_thread.start()
    sim_thread.start()
    panel_thread.start()
