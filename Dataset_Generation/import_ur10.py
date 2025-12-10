# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import math
import random
import time
import weakref

import omni
import omni.ui as ui
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.gui.components.ui_utils import btn_builder, get_style, setup_ui_headers
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics, UsdLux, UsdGeom, Usd, UsdShade

from . import domain_rng
from .common import set_drive_parameters
from isaacsim.core.prims import SingleArticulation



EXTENSION_NAME = "Import UR10"

"""
Revised the included Robotic Example Template
Could not figure out how to run a script any other way.
Added UI buttons to execute the functions I needed.
"""

path_default = "C:/Users/delta/Downloads"

class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        """Initialize extension and UI elements"""
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        self._ext_id = ext_id
        self._extension_path = ext_manager.get_extension_path(ext_id)
        self._window = None

        self.example_name = "UR10 URDF"
        self.category = "Import Robots"

        self.last_target_angles = []

        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=self._build_window,
            ui_hook=self._build_ui,
            category=self.category,
        )


    def _build_window(self):
        # self._window = omni.ui.Window(
        #     EXTENSION_NAME, width=0, height=0, visible=False, dockPreference=ui.DockPreference.LEFT_BOTTOM
        # )
        pass


    def _build_ui(self):

        with ui.VStack(spacing=5, height=0):

            title = "Import a UR10 via URDF"
            doc_link = "https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html"

            overview = "This Example shows you import a UR10 robot arm via URDF.\n\nPress the 'Open in IDE' button to view the source code."

            setup_ui_headers(self._ext_id, __file__, title, doc_link, overview)

            frame = ui.CollapsableFrame(
                title="Command Panel",
                height=0,
                collapsed=False,
                style=get_style(),
                style_type_name_override="CollapsableFrame",
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            )
            with frame:
                with ui.VStack(style=get_style(), spacing=5):
                    dict = {
                        "label": "Load Robot",
                        "type": "button",
                        "text": "Load",
                        "tooltip": "Load a UR10 Robot into the Scene",
                        "on_clicked_fn": self._on_load_robot,
                    }
                    btn_builder(**dict)

                    dict = {
                        "label": "Configure Drives",
                        "type": "button",
                        "text": "Configure",
                        "tooltip": "Configure Joint Drives",
                        "on_clicked_fn": self._on_config_robot,
                    }
                    btn_builder(**dict)

                    dict = {
                        "label": "Move to Pose",
                        "type": "button",
                        "text": "Move",
                        "tooltip": "Drive the Robot to a Random pose",
                        "on_clicked_fn": self.move_button,
                    }
                    btn_builder(**dict)

                    dict = {
                        "label": "Randomize Camera",
                        "type": "button",
                        "text": "Camera",
                        "tooltip": "Randomize the Camera Position",
                        "on_clicked_fn": self.camera_button,
                    }
                    btn_builder(**dict)

                    dict = {
                        "label": "Randomize Lights",
                        "type": "button",
                        "text": "Lights",
                        "tooltip": "Randomize the Lighting Conditions",
                        "on_clicked_fn": domain_rng.lights,
                    }
                    btn_builder(**dict)

                    dict = {
                        "label": "Randomize Textures",
                        "type": "button",
                        "text": "Textures",
                        "tooltip": "Randomize the",
                        "on_clicked_fn": self.random_tex,
                    }
                    btn_builder(**dict)

                    dict = {
                        "label": "Create Test Dataset",
                        "type": "button",
                        "text": "Test Set",
                        "tooltip": "Create Test Dataset",
                        "on_clicked_fn": self.test_button,
                    }
                    btn_builder(**dict)

                    dict = {
                        "label": "Create Train Dataset",
                        "type": "button",
                        "text": "Train Set",
                        "tooltip": "Create Train Dataset",
                        "on_clicked_fn": self.train_button,
                    }
                    btn_builder(**dict)

                    dict = {
                        "label": "log",
                        "type": "button",
                        "text": "log",
                        "tooltip": "Create Train Dataset",
                        "on_clicked_fn": self._log_robot_state,
                    }
                    btn_builder(**dict)

    def random_tex(self):
        domain_rng.randomize_floor()
        domain_rng.randomize_ur10()

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)
        self._window = None

    def _menu_callback(self):
        if self._window is None:
            self._build_ui()
        self._window.visible = not self._window.visible

    def _on_load_robot(self):
        load_stage = asyncio.ensure_future(omni.usd.get_context().new_stage_async())
        asyncio.ensure_future(self._load_robot(load_stage))

    async def _load_robot(self, task):
        done, pending = await asyncio.wait({task})
        if task in done:
            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
            import_config.merge_fixed_joints = False
            import_config.fix_base = True
            import_config.make_default_prim = True
            import_config.create_physics_scene = True
            omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=self._extension_path + "/data/urdf/robots/ur10/urdf/ur10.urdf",
                import_config=import_config,
            )

            # Default Camera Position
            viewport = get_active_viewport()
            if viewport:
                viewport.set_texture_resolution((1080, 1080))

            camera_state = ViewportCameraState("/OmniverseKit_Persp")
            camera_state.set_position_world(Gf.Vec3d(5.0, -2.0, 0.5), True)
            camera_state.set_target_world(Gf.Vec3d(0.0, 0.0, 0.0), True)

            # Default Physics, Gravity and Stage
            stage = omni.usd.get_context().get_stage()
            scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)

            # Ground Plane
            ground_path = "/World/Ground"
            gnd_size = 1000.0
            points = [
                Gf.Vec3f(-gnd_size/2, -gnd_size/2, -0.05),
                Gf.Vec3f( gnd_size/2, -gnd_size/2, -0.05),
                Gf.Vec3f( gnd_size/2,  gnd_size/2, -0.05),
                Gf.Vec3f(-gnd_size/2,  gnd_size/2, -0.05) ]
            counts = [4]
            indices = [0, 1, 2, 3]

            mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(ground_path))
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr(counts)
            mesh.CreateFaceVertexIndicesAttr(indices)

            # Add physics collision + rigid body to the plane
            UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(ground_path))
            UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(ground_path))
            UsdPhysics.RigidBodyAPI.Get(stage, ground_path).CreateRigidBodyEnabledAttr(False)

            # Default Lighting
            light_path = Sdf.Path("/Lights/DistantLight")
            distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path(light_path))
            distantLight.CreateIntensityAttr(5000)



    def _ensure_articulation(self):
        """Create and initialize the UR10 articulation wrapper."""
        stage = omni.usd.get_context().get_stage()

        prim = stage.GetPrimAtPath("/ur10")
        if not prim:
            print("[WARN] /ur10 prim not found yet, cannot init articulation.")
            return

        if not hasattr(self, "_articulation") or self._articulation is None:
            self._articulation = SingleArticulation(prim_path="/ur10", name="ur10")

        # Initialize
        try:
            self._articulation.initialize()
        except Exception as e:
            print(f"[WARN] Failed to initialize articulation: {e}")


    def _on_config_robot(self):
        stage = omni.usd.get_context().get_stage()

        PhysxSchema.PhysxArticulationAPI.Get(stage, "/ur10").CreateSolverPositionIterationCountAttr(64)
        PhysxSchema.PhysxArticulationAPI.Get(stage, "/ur10").CreateSolverVelocityIterationCountAttr(64)

        self.joint_1 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/ur10/joints/shoulder_pan_joint"), "angular")
        self.joint_2 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/ur10/joints/shoulder_lift_joint"), "angular")
        self.joint_3 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/ur10/joints/elbow_joint"), "angular")
        self.joint_4 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/ur10/joints/wrist_1_joint"), "angular")
        self.joint_5 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/ur10/joints/wrist_2_joint"), "angular")
        self.joint_6 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/ur10/joints/wrist_3_joint"), "angular")

        # Set the drive mode
        set_drive_parameters(self.joint_1, "position", math.degrees(0), math.radians(1e8), math.radians(5e7))
        set_drive_parameters(self.joint_2, "position", math.degrees(0), math.radians(1e8), math.radians(5e7))
        set_drive_parameters(self.joint_3, "position", math.degrees(0), math.radians(1e8), math.radians(5e7))
        set_drive_parameters(self.joint_4, "position", math.degrees(0), math.radians(1e8), math.radians(5e7))
        set_drive_parameters(self.joint_5, "position", math.degrees(0), math.radians(1e8), math.radians(5e7))
        set_drive_parameters(self.joint_6, "position", math.degrees(0), math.radians(1e8), math.radians(5e7))

        # Change stiffness of motion
        fast_stiffness = 5e4
        fast_damping = 1e3
        max_force = 1e6

        for drv in [self.joint_1, self.joint_2, self.joint_3,
                    self.joint_4, self.joint_5, self.joint_6]:
            drv.GetStiffnessAttr().Set(fast_stiffness)
            drv.GetDampingAttr().Set(fast_damping)
            drv.GetMaxForceAttr().Set(max_force)

        self._ensure_articulation()

    def move_button(self):
        asyncio.ensure_future(self.move_to_random())

    def camera_button(self):
        from omni.kit.viewport.utility import get_active_viewport_window
        import omni.usd
        from pxr import UsdGeom
        import numpy as np

        # Get active viewport
        viewport = get_active_viewport_window()
        if viewport is None:
            print("No active viewport window found")
            return

        # Get camera path used by this viewport (Sdf.Path)
        camera_path = viewport.viewport_api.camera_path
        print("Viewport camera path:", camera_path)

        # Get resolution (width, height)
        res = viewport.viewport_api.resolution
        width, height = int(res[0]), int(res[1])
        print("Viewport resolution:", width, height)

        # Get USD stage and camera prim
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(str(camera_path))  # cast Sdf.Path -> string

        if not camera_prim.IsValid():
            print(f"Camera prim at {camera_path} is not valid")
            return

        usd_cam = UsdGeom.Camera(camera_prim)

        # Focal length and sensor aperture from the camera
        focal_length = usd_cam.GetFocalLengthAttr().Get()  # in mm
        h_aperture = usd_cam.GetHorizontalApertureAttr().Get()  # in mm
        v_aperture = usd_cam.GetVerticalApertureAttr().Get()  # in mm

        print("Focal length (mm):", focal_length)
        print("Horizontal aperture (mm):", h_aperture)
        print("Vertical aperture (mm):", v_aperture)

        # Convert to pixel focal lengths
        fx = focal_length * width / h_aperture
        fy = focal_length * height / v_aperture
        cx = width / 2.0
        cy = height / 2.0

        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=float)

        print("Camera intrinsics matrix K:")
        print(K)


        asyncio.ensure_future(domain_rng.camera())


    async def move_to_random(self):
        print(f'[ROBOT] Moving to random position')

        collision = True
        while collision:

            neutral_pose = [0, -90, 0, 0, 0, 0]
            self.move_to_pose(neutral_pose)
            await asyncio.sleep(.5)

            # Generate Random Pose
            pose = [random.uniform(-180, 180),
                    random.uniform(-170, -10),
                    random.uniform(-170, 170),
                    random.uniform(-180, 180),
                    random.uniform(-180, 180),
                    random.uniform(-180, 180)]
            self.move_to_pose(pose)

            await asyncio.sleep(2)
            target, actual, _, _, _ = self.get_robot_state()

            if all(abs(t - a) < 0.1 for t, a in zip(target, actual)):
                collision = False
            else:
                print("[ROBOT] Collision detected, retrying")

        print(f"[ROBOT] Reached Pose: {[f'{p:.2f}' for p in pose]}")


    def move_to_pose(self, pose):
        self._on_config_robot()

        set_drive_parameters(self.joint_1, "position", pose[0])
        set_drive_parameters(self.joint_2, "position", pose[1])
        set_drive_parameters(self.joint_3, "position", pose[2])
        set_drive_parameters(self.joint_4, "position", pose[3])
        set_drive_parameters(self.joint_5, "position", pose[4])
        set_drive_parameters(self.joint_6, "position", pose[5])

        self.last_target_angles = pose


    async def capture_image(self, path = path_default, n =None):
        import os
        import time
        import asyncio
        from omni.kit.viewport.utility import (
            get_active_viewport,
            capture_viewport_to_file,
        )

        # Output Path
        save_dir = os.path.expanduser(path)
        os.makedirs(save_dir, exist_ok=True)
        if n is None: n = ''

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture{n}_{timestamp}.png"
        output_path = os.path.join(save_dir, filename)

        # Capture Image
        viewport = get_active_viewport()
        capture_helper = capture_viewport_to_file(viewport, file_path=output_path)

        # Wait for image to finish
        try:
            await capture_helper.wait_for_result(completion_frames=30)
            print(f"[IMAGE] Saved viewport image {n} --> {output_path}")
        except Exception as e:
            print(f"[ERROR] Capture failed: {e}")

    def _mat3_to_rpy_deg(self, m3):
        """Convert rotation matrix to RPY in degrees"""
        import math

        r00, r01, r02 = m3[0][0], m3[0][1], m3[0][2]
        r10, r11, r12 = m3[1][0], m3[1][1], m3[1][2]
        r20, r21, r22 = m3[2][0], m3[2][1], m3[2][2]

        if abs(r20) < 1.0:
            pitch = -math.asin(r20)
            roll = math.atan2(r21, r22)
            yaw = math.atan2(r10, r00)
        else:
            # Gimbal lock
            pitch = math.pi / 2 if r20 <= -1.0 else -math.pi / 2
            roll = math.atan2(-r12, r11)
            yaw = 0.0

        return (
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw),)

    def get_robot_state(self):
        """ Returns:
        joint_targets, joint_actual, joint_positions, joint_rotations, joint_2D_proj
        Only positions are in XYZ, the rest are degrees
        """

        stage = omni.usd.get_context().get_stage()
        joint_names = [
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]

        def safe_deg(value):
            if value is None:
                return float("nan")
            return math.degrees(value)

        # Target Input Angles (in degrees) from move command
        joint_targets = self.last_target_angles

        # Actual Joint Input Angles
        self._ensure_articulation()
        joint_actual = []

        if hasattr(self, "_articulation") and self._articulation is not None:
            positions_rad = self._articulation.get_joint_positions()

            if positions_rad is None:
                print("[WARN] get_joint_positions() returned None (sim not playing yet?)")
                joint_actual = [float("nan")] * 6
            else:
                joint_actual = [math.degrees(float(p)) for p in positions_rad]
        else:
            joint_actual= [float("nan")] * 6

        # XYZ Position of Joint
        joint_positions = {}
        joint_rotations = {}
        joint_2d_proj = {}

        for j in joint_names:
            joint_path = f"/ur10/{j}"
            prim = stage.GetPrimAtPath(joint_path)
            if not prim:
                continue

            xformable = UsdGeom.Xformable(prim)
            world_mat = xformable.ComputeLocalToWorldTransform(0.0)

            pos = world_mat.ExtractTranslation()
            joint_positions[j] = (float(pos[0]), float(pos[1]), float(pos[2]))

            # RPY Euler Angle Rotations
            R = Gf.Matrix3d(
                world_mat[0][0], world_mat[0][1], world_mat[0][2],
                world_mat[1][0], world_mat[1][1], world_mat[1][2],
                world_mat[2][0], world_mat[2][1], world_mat[2][2])
            roll, pitch, yaw = self._mat3_to_rpy_deg(R)
            joint_rotations[j] = (float(roll), float(pitch), float(yaw))

            # 2D projections
            u, v, d= self._world_to_screen(joint_positions[j])
            joint_2d_proj[j] = (u, v, d)

        return joint_targets, joint_actual, joint_positions, joint_rotations, joint_2d_proj

    def _log_robot_state(self,path = path_default, command_name: str = "", state = None):
        import os
        import csv
        import time

        # Get Current state if none given.
        if state is None:
            joint_targets, joint_actual, joint_positions, joint_rotations, joint_2D_proj = self.get_robot_state()
        else:
            joint_targets, joint_actual, joint_positions, joint_rotations, joint_2D_proj = state

        # Where to save
        save_dir = os.path.expanduser(path)
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "robot_state.csv")

        # Fixed order for joints and links to keep columns consistent
        joint_indices = [0, 1, 2, 3, 4, 5]

        joint_names = [
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]
        joint_labels = [
            "J0_Base",
            "J1_Shoulder",
            "J2_Elbow",
            "J3_Wrist",
            "J4_Wrist",
            "J5_Wrist"]

        # Build header if file does not exist
        write_header = not os.path.exists(log_path)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)

            # Write Data Headers
            if write_header:
                header = ["timestamp", "command"]

                for j in joint_indices:
                    header.append(f"J{j}_target")

                for j in joint_indices:
                    header.append(f"J{j}_actual")

                for jn in joint_labels:
                    header.extend([f"{jn}_x", f"{jn}_y", f"{jn}_z"])
                    header.extend([f"{jn}_Roll", f"{jn}_Pitch", f"{jn}_Yaw"])

                for j in joint_labels:
                    header.extend([f"2D_{j}_x", f"2D_{j}_y", f"2D_{j}_dist"])

                writer.writerow(header)

            # Add data from sample
            row = [timestamp, command_name]
            row.extend(joint_targets)
            row.extend(joint_actual)

            # Joint XYZ, Rotations, and 2D projection
            for jn in joint_names:
                if jn in joint_positions:
                    x, y, z = joint_positions[jn]
                    roll, pitch, yaw = joint_rotations[jn]
                else:
                    x, y, z = float("nan"), float("nan"), float("nan")
                    roll, pitch, yaw = float("nan"), float("nan"), float("nan")

                row.extend([float(x), float(y), float(z)])
                row.extend([float(roll), float(pitch), float(yaw)])

            # 2D Joint projections
            for jn in joint_names:
                x_2D, y_2D, d_2D = joint_2D_proj[jn]
                try:
                    row.extend([float(x_2D), float(y_2D), float(d_2D)])
                except TypeError:
                    print(joint_2D_proj)
                    row.extend([float("nan"), float("nan"), float(d_2D)])

            writer.writerow(row)

        print(f"[LOG] Appended robot state {command_name} to {log_path}")

    def _world_to_screen(self, world_pos):
        """Project a world-space 3D point to 2D pixel coordinates in the active viewport.

        Returns (u, v) in pixel coordinates, or (None, None) if not visible.
        """
        from pxr import Gf
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        # Error Handling
        if viewport is None:
            print("[WARN] No active viewport")
            return None, None, None

        if not isinstance(world_pos, Gf.Vec3d):
            world_pos = Gf.Vec3d(*world_pos)

        # Get Camera distance to point
        stage = omni.usd.get_context().get_stage()
        cam_path = viewport.get_active_camera()
        cam_prim = stage.GetPrimAtPath(cam_path)

        cam_xform = UsdGeom.Xformable(cam_prim)
        cam_world = cam_xform.ComputeLocalToWorldTransform(0.0)
        cam_pos = cam_world.ExtractTranslation()

        dist = (world_pos - cam_pos).GetLength()

        # Transform point to 2D camera projection
        ndc_vec = viewport.world_to_ndc.Transform(world_pos)
        ndc_x, ndc_y, ndc_z = ndc_vec[0], ndc_vec[1], ndc_vec[2]

        # If out of frame.
        if abs(ndc_x) > 1.0 or abs(ndc_y) > 1.0:
            return None, None, float(dist)

        (u, v), in_viewport = viewport.map_ndc_to_texture_pixel((ndc_x, ndc_y))
        if not in_viewport:
            return None, None, float(dist)

        return float(u), float(v), float(dist)

    def test_button(self):
        asyncio.ensure_future(self.create_test_set())

    def train_button(self):
        asyncio.ensure_future(self.create_train_set())

    async def create_test_set(self):

        N = 2000
        path = path_default + "/Test"
        print(f"[INFO] Creating Test Set of N={N}")

        start_time = time.time()

        for n in range(1, N + 1):
            iter_start = time.time()

            #domain_rng.textures.randomize_ur10()
            #domain_rng.textures.randomize_floor()
            #domain_rng.lights()

            await self.move_to_random()
            await domain_rng.camera()
            await asyncio.sleep(0.1)

            # Revise Camera if out of screen
            state = self.get_robot_state()
            *_, joint_2D = state
            while any(u is None or v is None for (u, v, d) in joint_2D.values()):
                print(f"[CAMERA] Robot Out of Frame. Revising")
                await domain_rng.camera()
                await asyncio.sleep(0.1)
                state = self.get_robot_state()
                *_, joint_2D = state

            await asyncio.sleep(0.5)
            self._log_robot_state(path, f'Sample {n}', state=state)
            await self.capture_image(path, n)
            await asyncio.sleep(1.0)

            # Track Time
            elapsed_total = time.time() - start_time
            iter_time = time.time() - iter_start
            remaining = (N - n) * iter_time

            print(
                f"[{n}/{N}]  ({((n / N) * 100):.2f}%) | "
                f"Iter: {iter_time:4.2f}s | "
                f"Elapsed: {(elapsed_total/60):6.2f}m | "
                f"ETA: {(remaining / 60):6.2f}m"
            )

    async def create_train_set(self):

        N = 5000
        path = path_default + "/Train"
        print(f"[INFO] Creating Train Set of N={N}")

        start_time = time.time()

        for n in range(1, N + 1):
            iter_start = time.time()

            domain_rng.textures.randomize_ur10()
            domain_rng.textures.randomize_floor()
            domain_rng.lights()

            await self.move_to_random()
            await domain_rng.camera()
            await asyncio.sleep(0.1)

            # Revise Camera if out of screen
            state = self.get_robot_state()
            *_, joint_2D = state
            while any(u is None or v is None for (u, v, d) in joint_2D.values()):
                print(f"[CAMERA] Robot Out of Frame. Revising")
                await domain_rng.camera()
                await asyncio.sleep(0.1)
                state = self.get_robot_state()
                *_, joint_2D = state

            await asyncio.sleep(0.5)
            self._log_robot_state(path, f'Sample {n}', state=state)
            await self.capture_image(path, n)
            await asyncio.sleep(1.0)

            # Track Time
            elapsed_total = time.time() - start_time
            iter_time = time.time() - iter_start
            remaining = (N - n) * iter_time

            print(
                f"[{n}/{N}]  ({((n / N) * 100):.2f}%) | "
                f"Iter: {iter_time:4.2f}s | "
                f"Elapsed: {elapsed_total:6.2f}s | "
                f"ETA: {(remaining / 60):6.2f}m"
            )

