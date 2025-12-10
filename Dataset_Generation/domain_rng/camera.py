import math
import random
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf


# Randomize Camera Orientation and Position
async def camera(max_R = 8.0):
    camera_state = ViewportCameraState("/OmniverseKit_Persp")

    # Random camera focal offset
    z_offset = random.uniform(-1, 1)
    x_offset = random.uniform(-1, 1)
    y_offset = random.uniform(-1, 1)

    # Spherical Polar Coordinates
    min_radius, max_radius = 2.5, max_R
    r = random.uniform(min_radius, max_radius)

    min_deg, max_deg = 10.0, 80.0
    angle = math.radians(random.uniform(min_deg, max_deg))
    rotation = random.uniform(-1* math.pi, math.pi)

    # Back to Cartesian
    x = (r * math.cos(angle) * math.cos(rotation) )
    y = (r * math.cos(angle) * math.sin(rotation) )
    z = (r * math.sin(angle))

    # Set camera position and make it look at the random spot
    camera_state.set_position_world(Gf.Vec3d(x, y, z), True)
    camera_state.set_target_world(Gf.Vec3d(x_offset, y_offset,z_offset), True)

    print(
        f"[CAMERA] Randomized camera: r={r:.2f}, {math.degrees(angle):.1f}°, "
        f"{math.degrees(rotation):.1f}°, Facing: ({x_offset:.2f}, {y_offset:.2f}, {z_offset:.2f})"
    )
