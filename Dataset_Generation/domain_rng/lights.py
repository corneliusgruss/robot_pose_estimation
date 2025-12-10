import random
import omni
from pxr import Gf, Sdf, UsdLux, UsdLux, UsdGeom, Usd, UsdShade


# Randomize Amount, Placement and Intensity of Lights
def lights():
    stage = omni.usd.get_context().get_stage()
    lights_root_path = Sdf.Path("/Lights")

    if not stage.GetPrimAtPath(lights_root_path):
        stage.DefinePrim(lights_root_path, "Xform")

    lights_root = stage.GetPrimAtPath(lights_root_path)

    # Delete Previous Lighting
    for child in list(lights_root.GetChildren()):
        stage.RemovePrim(child.GetPath())

    # Random Amount of Lights
    num_lights = random.randint(1, 5)  # change range as you like

    for i in range(num_lights):
        light_path = lights_root_path.AppendChild(f"SphereLight_{i}")
        sphere_light = UsdLux.SphereLight.Define(stage, light_path)

        xform = UsdGeom.Xformable(sphere_light)

        # Random position above the origin
        x = random.uniform(-3.0, 3.0)
        y = random.uniform(-3.0, 3.0)
        z = random.uniform(2.0, 6.0)
        xform.AddTranslateOp().Set(Gf.Vec3f(x, y, z))

        # Random radius
        radius = random.uniform(0.1, 0.5)
        sphere_light.CreateRadiusAttr(radius)

        # Random color and inensity
        r = random.uniform(0.15, 1.0)
        g = random.uniform(0.15, 1.0)
        b = random.uniform(0.15, 1.0)
        sphere_light.CreateColorAttr(Gf.Vec3f(r, g, b))

        intensity = random.uniform(50_000, 300_000)
        sphere_light.CreateIntensityAttr(intensity)

    print(f'[LIGHTS] Lights Randomized, {num_lights} total')
