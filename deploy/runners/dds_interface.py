import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from utils.joint_mapper import URDFPolicyMapper


class DDSInterface:
    """Low-level DDS interface for communicating with robot/simulator."""

    def __init__(self, joint_mapper: URDFPolicyMapper):
        self.joint_mapper = joint_mapper
        self.latest_state = None

        # DDS setup
        ChannelFactoryInitialize(1, "lo")
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._state_callback, 10)
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.crc = CRC()

    def _state_callback(self, msg: LowState_):
        """Callback for receiving robot state updates."""
        self.latest_state = msg

    def get_state(self):
        """Get current robot state in policy joint order."""
        if self.latest_state is None:
            return None

        # Extract joint positions and velocities in URDF order
        dof_pos_urdf = torch.tensor(
            [self.latest_state.motor_state[i].q for i in range(12)],
            dtype=torch.float32
        )
        dof_vel_urdf = torch.tensor(
            [self.latest_state.motor_state[i].dq for i in range(12)],
            dtype=torch.float32
        )

        # Extract IMU data
        base_ang_vel = torch.tensor(
            self.latest_state.imu_state.gyroscope,
            dtype=torch.float32
        )
        # Quaternion to projected gravity
        quat = self.latest_state.imu_state.quaternion  # w, x, y, z
        projected_gravity = self._get_projected_gravity(quat)

        # Convert to policy order
        dof_pos = self.joint_mapper.to_policy_order(dof_pos_urdf)
        dof_vel = self.joint_mapper.to_policy_order(dof_vel_urdf)

        return {
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
        }

    def _quat_rotate_inverse(self, q, v):
        """Rotate vector by inverse of quaternion (w,x,y,z format)."""
        w, x, y, z = q
        # Conjugate for inverse rotation
        qw, qx, qy, qz = w, -x, -y, -z
        vx, vy, vz = v
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        return torch.tensor([
            vx + qw * tx + qy * tz - qz * ty,
            vy + qw * ty + qz * tx - qx * tz,
            vz + qw * tz + qx * ty - qy * tx,
        ], dtype=torch.float32)

    def _get_projected_gravity(self, quat):
        """Get gravity vector in body frame."""
        return self._quat_rotate_inverse(quat, [0.0, 0.0, -1.0])

    def send_command(self, target_policy, kp, kd):
        """Send command to robot. target_policy is in policy joint order."""
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        # Convert from policy order to URDF order for the robot
        target_urdf = self.joint_mapper.to_urdf_order(target_policy)

        for idx in range(12):
            cmd.motor_cmd[idx].mode = 0x01
            cmd.motor_cmd[idx].q = float(target_urdf[idx])
            cmd.motor_cmd[idx].kp = float(kp)
            cmd.motor_cmd[idx].dq = 0.0
            cmd.motor_cmd[idx].kd = float(kd)
            cmd.motor_cmd[idx].tau = 0.0

        cmd.crc = self.crc.Crc(cmd)
        self.pub.Write(cmd)
