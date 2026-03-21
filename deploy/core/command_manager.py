import threading
import time


class CommandManager:
    """Handles velocity command input for the robot controller."""

    def __init__(self, controller):
        self.controller = controller
        self.running = False
        self._thread = None

        # Default commands
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_z = 0.0

    def set_commands(self, lin_vel_x: float = 0.0, lin_vel_y: float = 0.0, ang_vel_z: float = 0.0):
        """Set velocity commands."""
        self.lin_vel_x = lin_vel_x
        self.lin_vel_y = lin_vel_y
        self.ang_vel_z = ang_vel_z
        self.controller.set_commands(
            lin_vel_x=lin_vel_x,
            lin_vel_y=lin_vel_y,
            ang_vel_z=ang_vel_z
        )

    def start_keyboard_control(self):
        """Start a thread for keyboard velocity control."""
        self.running = True
        self._thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._thread.start()
        print("[CommandManager] Keyboard control started.")
        print("  W/S: Forward/Backward")
        print("  A/D: Left/Right strafe")
        print("  Q/E: Turn left/right")
        print("  Space: Stop")

    def stop(self):
        """Stop keyboard control thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _keyboard_loop(self):
        """Non-blocking keyboard input loop."""
        try:
            import sys
            import termios
            import tty

            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                while self.running:
                    if self._kbhit():
                        key = sys.stdin.read(1).lower()
                        self._handle_key(key)
                    time.sleep(0.05)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except ImportError:
            print("[CommandManager] Keyboard control not available on this platform.")

    def _kbhit(self):
        """Check if keyboard input is available."""
        import select
        import sys
        return select.select([sys.stdin], [], [], 0)[0] != []

    def _handle_key(self, key: str):
        """Handle keyboard input."""
        speed = 0.5
        turn_speed = 0.5

        if key == 'w':
            self.lin_vel_x = min(self.lin_vel_x + speed, 1.5)
        elif key == 's':
            self.lin_vel_x = max(self.lin_vel_x - speed, -1.0)
        elif key == 'a':
            self.lin_vel_y = min(self.lin_vel_y + speed, 0.5)
        elif key == 'd':
            self.lin_vel_y = max(self.lin_vel_y - speed, -0.5)
        elif key == 'q':
            self.ang_vel_z = min(self.ang_vel_z + turn_speed, 1.0)
        elif key == 'e':
            self.ang_vel_z = max(self.ang_vel_z - turn_speed, -1.0)
        elif key == ' ':
            self.lin_vel_x = 0.0
            self.lin_vel_y = 0.0
            self.ang_vel_z = 0.0

        self.controller.set_commands(
            lin_vel_x=self.lin_vel_x,
            lin_vel_y=self.lin_vel_y,
            ang_vel_z=self.ang_vel_z
        )
        print(f"\r[Cmd] vx={self.lin_vel_x:.2f} vy={self.lin_vel_y:.2f} wz={self.ang_vel_z:.2f}  ", end="")
