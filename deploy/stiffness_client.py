"""Send a runtime stiffness_commands update to a running quadruped-policy container.

Usage:
    python deploy/stiffness_client.py --stiffness 500.0
    python deploy/stiffness_client.py --stiffness 1000.0 --port 7777 --host localhost
"""

import argparse
import json
import socket
import sys


def main():
    parser = argparse.ArgumentParser(description="Update stiffness_commands in a live policy run")
    parser.add_argument("--stiffness", type=float, required=True,
                        help="New stiffness_commands value to send to the policy")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Host where the policy container is running (default: localhost)")
    parser.add_argument("--port", type=int, default=7777,
                        help="CommandServer TCP port (default: 7777)")
    args = parser.parse_args()

    payload = json.dumps({"stiffness_commands": args.stiffness}) + "\n"

    try:
        with socket.create_connection((args.host, args.port), timeout=5.0) as sock:
            sock.sendall(payload.encode())
            response = sock.makefile().readline()
    except ConnectionRefusedError:
        print(f"[ERROR] Connection refused at {args.host}:{args.port} — is the policy running?",
              file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        print(f"[ERROR] Unexpected response: {response!r}", file=sys.stderr)
        sys.exit(1)

    if result.get("status") == "ok":
        print(f"[OK] stiffness_commands set to {result['stiffness_commands']}")
    else:
        print(f"[ERROR] {result.get('message', result)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
