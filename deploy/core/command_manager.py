"""Keyboard input handler for live velocity control.

Runs in a background thread.
Fully self-contained — no dependency on controller or deployer internals.

Usage:
    commands = {"velocity_commands": np.array([0.5, 0.0, 0.0])}
    kb = KeyboardController(commands["velocity_commands"])
    kb.start()
    # ... control loop reads commands["velocity_commands"] each step ...
    kb.stop()
"""

import sys
import threading
import time

import numpy as np


class KeyboardController:
    """Background thread that maps WASD keys to velocity command updates."""

    # Arrow keys send escape sequences: ESC [ A/B/C/D
    # We handle them in _on_key via _read_escape_seq
    KEYMAP = {
        "i": (0, +1),   # forward
        "k": (0, -1),   # backward
        "j": (1, +1),   # strafe left
        "l": (1, -1),   # strafe right
        "u": (2, +1),   # turn left
        "o": (2, -1),   # turn right
        # Arrow keys (resolved from escape sequences)
        "UP":    (0, +1),   # forward
        "DOWN":  (0, -1),   # backward
        "LEFT":  (2, +1),   # turn left
        "RIGHT": (2, -1),   # turn right
    }

    LIMITS = np.array([
        [-1.0, 1.5],   # lin_vel_x
        [-0.5, 0.5],   # lin_vel_y
        [-1.0, 1.0],   # ang_vel_z
    ], dtype=np.float32)

    LABELS = ("vx", "vy", "wz")

    def __init__(
        self,
        velocity_array: np.ndarray,
        step_lin: float = 0.2,
        step_ang: float = 0.2,
    ):
        self._vel = velocity_array
        self._steps = np.array([step_lin, step_lin, step_ang], dtype=np.float32)
        self._running = False
        self._thread = None
        self._old_term = None


    def start(self):
        """Start listening for keyboard input."""
        try:
            import termios
            import tty
            self._old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except (ImportError, Exception) as exc:
            print(f"[KeyboardController] Not available: {exc}")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[KeyboardController] Keyboard control active")
        print("  Arrows: forward/backward/turn    I/K: forward/backward")
        print("  J/L: strafe left/right            U/O: turn left/right")
        print("  Space: stop all")
        return True

    def stop(self):
        """Stop the input thread and restore terminal."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._restore_terminal()


    def _restore_terminal(self):
        if self._old_term is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)
            self._old_term = None

    def _loop(self):
        import select
        try:
            while self._running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    key = self._resolve_key(ch, select)
                    self._on_key(key)
        finally:
            self._restore_terminal()

    def _resolve_key(self, ch, select):
        """Resolve escape sequences (arrow keys) to named keys."""
        if ch == "\x1b":  # ESC
            if select.select([sys.stdin], [], [], 0.01)[0]:
                ch2 = sys.stdin.read(1)
                if ch2 == "[" and select.select([sys.stdin], [], [], 0.01)[0]:
                    ch3 = sys.stdin.read(1)
                    return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(ch3, "")
            return ""
        return ch.lower()

    def _on_key(self, key: str):
        if key == " ":
            self._vel[:] = 0.0
        elif key in self.KEYMAP:
            axis, sign = self.KEYMAP[key]
            self._vel[axis] += sign * self._steps[axis]
            self._vel[axis] = np.clip(
                self._vel[axis], self.LIMITS[axis, 0], self.LIMITS[axis, 1]
            )
        else:
            return

        status = "  ".join(
            f"{lbl}={self._vel[i]:+.2f}" for i, lbl in enumerate(self.LABELS)
        )
        print(f"\r[CMD] {status}          ", end="\r")
