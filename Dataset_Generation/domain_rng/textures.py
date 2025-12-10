import os
import random

import omni.usd
from pxr import Sdf, Gf, UsdShade, UsdGeom

#UR10 Textures
path = r'C:/isaac-sim/exts/isaacsim.asset.importer.urdf/isaacsim/asset/importer/urdf/impl/samples/textures/Base'
UR10_TEXTURE_DIRS = [
    os.path.join(path, "Carpet/Carpet_Diamond_Yellow"),
    os.path.join(path, "Masonry/Brick_Pavers"),
    os.path.join(path, "Metals/Aluminum_Anodized"),
    os.path.join(path, "Metals/Aluminum_Cast"),
    os.path.join(path, "Metals/Brass"),
    os.path.join(path, "Metals/Brushed_Antique_Copper"),
    os.path.join(path, "Metals/Copper"),
    os.path.join(path, "Metals/Steel_Stainless"),
    os.path.join(path, "Plastics/Rubber_Smooth"),
    os.path.join(path, "Plastics/Rubber_Textured"),
]
PATTERN_DIR = r'C:\isaac-sim\exts\isaacsim.asset.importer.urdf\isaacsim\asset\importer\urdf\impl\samples\textures\Patterns'

# Ground Plane Textures


# Make material from file paths
def _build_material_from_textures(stage, texture_dir, material_path):

    if stage is None:
        print("[TEXTURES] No stage – cannot build material.")
        return None

    if not os.path.isdir(texture_dir):
        print(f"[TEXTURES] Texture folder does not exist: {texture_dir}")
        return None

    files = os.listdir(texture_dir)

    basecolor = None
    normal = None

    for f in files:
        lf = f.lower()
        if not lf.endswith(".png"):
            continue

        if "basecolor" in lf:
            basecolor = f
        elif lf.endswith("_n.png") or "normal" in lf:
            normal = f

    if basecolor is None:
        print(f"[TEXTURES] ERROR: No BaseColor texture in {texture_dir}")
        return None

    basecolor_path = os.path.join(texture_dir, basecolor).replace("\\", "/")
    normal_path = (
        os.path.join(texture_dir, normal).replace("\\", "/")
        if normal
        else None
    )

    print(f"[TEXTURES] BaseColor: {os.path.basename(basecolor_path)}")
    if normal_path:
        print(f"[TEXTURES] Normal:    {os.path.basename(normal_path)}")

    # -- Create material prim --
    mat = UsdShade.Material.Define(stage, Sdf.Path(material_path))

    # PreviewSurface shader
    shader_path = Sdf.Path(material_path).AppendPath("Shader")
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    # Primvar reader for UVs ("st")
    st_reader_path = Sdf.Path(material_path).AppendPath("stReader")
    st_reader = UsdShade.Shader.Define(stage, st_reader_path)
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.TexCoord2f)

    # BaseColor texture node
    base_tex_path = Sdf.Path(material_path).AppendPath("BaseColorTex")
    base_tex = UsdShade.Shader.Define(stage, base_tex_path)
    base_tex.CreateIdAttr("UsdUVTexture")
    base_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(basecolor_path)
    )
    base_tex.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(st_out)
    base_tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    base_tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    base_out = base_tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        base_out
    )

    # Normal map
    if normal_path:
        normal_tex_path = Sdf.Path(material_path).AppendPath("NormalTex")
        normal_tex = UsdShade.Shader.Define(stage, normal_tex_path)
        normal_tex.CreateIdAttr("UsdUVTexture")
        normal_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(normal_path)
        )
        normal_tex.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(
            st_out
        )
        normal_tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        normal_tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        norm_out = normal_tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            norm_out
        )

    # Some default PBR parameters
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    # Connect shader -> material surface
    surf_out = mat.CreateSurfaceOutput()
    shader_out = shader.GetOutput("out")
    if not shader_out:
        shader_out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    surf_out.ConnectToSource(shader_out)

    return mat

# Bind material to all links in prim
def _apply_material_to_visuals(stage, material, visuals_root="/visuals"):

    if stage is None:
        print("[textures] No stage – cannot apply material.")
        return 0

    root = stage.GetPrimAtPath(visuals_root)
    if not root or not root.IsValid():
        print(f"[textures] ERROR: {visuals_root} not found.")
        return 0

    mesh_types = {"Mesh", "Cube", "Cylinder", "Sphere", "Cone", "Capsule"}
    count = 0

    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(visuals_root):
            continue
        if prim.GetTypeName() not in mesh_types and not prim.GetName().endswith("_link"):
            continue

        api = UsdShade.MaterialBindingAPI(prim)

        # Clear existing bindings
        try:
            api.UnbindAllBindings()
        except Exception:
            rel = api.GetDirectBindingRel()
            if rel:
                rel.ClearTargets()

        # Bind with strong strength so this wins over descendants
        api.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        count += 1

    return count

# Radomize Texture of Robotic Arm
def randomize_ur10(texture_dir=None, visuals_root="/visuals"):

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[TEXTURES] No stage – cannot randomize UR10.")
        return

    # Ensure /Looks exists
    looks_path = Sdf.Path("/Looks")
    if not stage.GetPrimAtPath(looks_path):
        UsdGeom.Scope.Define(stage, looks_path)

    # Choose a texture folder
    if texture_dir is None:
        valid_dirs = [d for d in UR10_TEXTURE_DIRS if os.path.isdir(d)]
        if not valid_dirs:
            print("[TEXTURES] No valid UR10_TEXTURE_DIRS configured.")
            return
        texture_dir = random.choice(valid_dirs)

    mat_path = "/Looks/UR10_Textured"

    mat = _build_material_from_textures(stage, texture_dir, mat_path)
    if mat is None:
        print("[Error] UR10 Texture Failed.")
        return

    count = _apply_material_to_visuals(stage, mat, visuals_root)
    print(f"[TEXTURE] Bound UR10_Textured to {count} prims under {visuals_root}.")

# Ramdomize Floor, based on image or not
def randomize_floor(prim_path="/World/Ground", use_image_prob=0.7):

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[TEXTURES] No stage – cannot randomize floor.")
        return

    ground_prim = stage.GetPrimAtPath(prim_path)
    if not ground_prim or not ground_prim.IsValid():
        print(f"[TEXTURES] Prim {prim_path} not found – cannot randomize floor.")
        return

    ground_mesh = UsdGeom.Mesh(ground_prim)
    if not ground_mesh:
        print(f"[TEXTURES] Prim {prim_path} is not a Mesh – cannot randomize floor.")
        return

    # Make sure it's visible
    UsdGeom.Imageable(ground_prim).MakeVisible()

    # ---------------------------------------------------------
    # Ensure 'st' UV primvar on the mesh, so textures can use it
    # ---------------------------------------------------------
    primvars_api = UsdGeom.PrimvarsAPI(ground_prim)
    st_primvar = primvars_api.GetPrimvar("st")
    if not st_primvar:
        st_primvar = primvars_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )

    points = ground_mesh.GetPointsAttr().Get()
    num_points = len(points) if points is not None else 0

    if num_points > 0:

        world_units_per_tile = random.uniform(1.0, 10.0)
        st_values = []
        for p in points:
            x, y = p[0], p[1]

            u = (x / world_units_per_tile) + 0.5
            v = (y / world_units_per_tile) + 0.5
            st_values.append(Gf.Vec2f(u, v))

        st_primvar.Set(st_values)
    else:
        st_primvar.Set([])


    # ---------------------------------------------------------
    # Build / reuse a material in /World/Looks/Floor_Textured
    # ---------------------------------------------------------
    looks_path = Sdf.Path("/World/Looks")
    if not stage.GetPrimAtPath(looks_path):
        stage.DefinePrim(looks_path, "Scope")

    mat_path = looks_path.AppendPath("Floor_Textured")

    # If a material already exists here, remove it (along with its
    # stReader / DiffuseTex nodes and any old connections).
    existing_mat_prim = stage.GetPrimAtPath(mat_path)
    if existing_mat_prim and existing_mat_prim.IsValid():
        stage.RemovePrim(mat_path)

    mat = UsdShade.Material.Define(stage, mat_path)

    shader_path = mat_path.AppendPath("Shader")
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    # Random base color and PBR params
    base_color = Gf.Vec3f(
        random.uniform(0.2, 0.9),
        random.uniform(0.2, 0.9),
        random.uniform(0.2, 0.9),
    )
    roughness = random.uniform(0.2, 0.9)
    sheen_weight = random.uniform(0.0, 0.4)

    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("sheenWeight", Sdf.ValueTypeNames.Float).Set(sheen_weight)
    shader.CreateInput("sheenColor", Sdf.ValueTypeNames.Color3f).Set(base_color)

    diffuse_in = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)

    # ---------------------------------------------------------
    # Decide if we use an image texture
    # ---------------------------------------------------------
    use_image = (random.random() < use_image_prob)
    tex_out = None

    if use_image and os.path.isdir(PATTERN_DIR):
        valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".bmp", ".dds")
        pattern_files = [
            f for f in os.listdir(PATTERN_DIR)
            if f.lower().endswith(valid_exts)
        ]

        if pattern_files:
            pattern_file = random.choice(pattern_files)
            pattern_path = os.path.join(PATTERN_DIR, pattern_file).replace("\\", "/")
            print(f"[TEXTURES] Floor: using pattern image {os.path.basename(pattern_path)}")

            # Primvar reader for 'st'
            st_reader_path = mat_path.AppendPath("stReader")
            st_reader = UsdShade.Shader.Define(stage, st_reader_path)
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.TexCoord2f)

            # UV Texture node
            tex_path = mat_path.AppendPath("DiffuseTex")
            tex = UsdShade.Shader.Define(stage, tex_path)
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(pattern_path))
            tex.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(st_out)
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        else:
            print("[TEXTURES] Floor: no images found in PATTERN_DIR – falling back to flat color.")
            use_image = False
    else:
        if use_image:
            print(f"[TEXTURES] PATTERN_DIR does not exist: {PATTERN_DIR} – falling back to flat color.")
        use_image = False

    # Connect either the texture or the flat color
    if use_image and tex_out is not None:
        diffuse_in.ConnectToSource(tex_out)
    else:
        diffuse_in.Set(base_color)

    # Output and bind material
    surf_out = mat.CreateSurfaceOutput()
    shader_out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    surf_out.ConnectToSource(shader_out)

    api = UsdShade.MaterialBindingAPI(ground_prim)
    try:
        api.UnbindAllBindings()
    except Exception:
        rel = api.GetDirectBindingRel()
        if rel:
            rel.ClearTargets()
    api.Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    print(f"[TEXTURES] Bound Floor_Textured to {ground_prim.GetPath()} (use_image={use_image})")
