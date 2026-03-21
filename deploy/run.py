from pathlib import Path
from core.policy_controller import PolicyController
from core.command_manager import CommandManager
from core.config import Config
from runners.dds_runner import DDSRunner

POLICY_DIR = Path(__file__).parent / "policies" / "unitree_go2_walk_2026-03-05"


def main():
    policy_path = POLICY_DIR / "policy.pt"
    config_path = POLICY_DIR / "config.yaml"

    # Load policy config from YAML
    policy_config = Config.from_yaml(str(config_path))

    # Initialize controller with policy and config
    controller = PolicyController(str(policy_path), policy_config)

    # Initialize command manager for velocity commands
    command_manager = CommandManager(controller)

    # Set initial commands (forward walk)
    command_manager.set_commands(lin_vel_x=0.5, lin_vel_y=0.0, ang_vel_z=0.0)

    # Optional: Enable keyboard control
    # command_manager.start_keyboard_control()

    # Initialize and run DDS runner
    runner = DDSRunner(controller, policy_config)
    runner.run()


if __name__ == "__main__":
    main()
