ROBOT = "go2" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1"
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene.xml" # Robot scene
DOMAIN_ID = 1 # Domain id
INTERFACE = "lo" # Interface

USE_JOYSTICK = 0 # Simulate Unitree WirelessController using a gamepad
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False # Virtual spring band, used for lifting h1
ENABLE_VERTICAL_PERTURBATION = True # Periodic vertical force on torso (Space to toggle)
ENABLE_LATERAL_PERTURBATION = True # Random lateral (XY) pushes on torso (P to toggle)
ENABLE_DIRECTIONAL_PERTURBATION = True # Arrow-key controlled XY force on torso
ENABLE_STIFFNESS_KEYS = True          # [ / ] keys to decrease / increase stiffness_commands
STIFFNESS_KEY_STEP = 100.0            # how much each keypress changes stiffness_commands

SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer

# Camera initial view (applied once at startup)
CAM_AZIMUTH   = 90 # 135    # horizontal orbit angle, degrees (0 = front)
CAM_ELEVATION  = 0 # -20   # vertical tilt, degrees (negative = looking down)
CAM_DISTANCE   = 2.0   # distance from lookat point, metres
CAM_LOOKAT     = [0.05, 0.0, 0.25] # 0.3] # 15] # [0.0, 0.0, 0.3]  # point the camera orbits around
