from my_utils import *
from set_interface_try import *
from partnet_skill_common import largest_abs_joint, repose_waypoints
import random

# Was a fixed 10 deg yaw. Sample once per process so the real microwave,
# the concept, and the hand stay on the same angle.
random_degree = float(np.random.uniform(-20.0, 20.0))
rot_real, random_degree = rotate_z(
            current_quat=[1, 0, 0, 0],
            angle_deg=random_degree,
        )
print(f"[microwave] spawn yaw={random_degree:.1f} deg")
class mircrowave ():
    def __init__(self,):
        self.scale_micwave = (0.3,0.3,0.3)
        self.concept_urdf_path = None
        self.hand_rot = None
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs 
    ):
        self.num_envs = num_envs
        concept_root = str(PARTNET_CONCEPT_ROOT)
        category = "Microwave"
        concept_urdf_fallback= str(PARTNET_CONCEPT_ROOT / "Microwave" / "87bae84777fe8b702bac1bcdfc2402d2" / "Microwave.urdf")


        # rot_real, self.random_degree = rotate_z(
        #     current_quat=[1, 0, 0, 0],
        #     angle_deg=float(random.uniform(40, 50.0)),
        # )
        rot_concept = rotate_x(current_quat=[0.707, 0, 0, -0.707], angle_deg=random_degree)
        # self.hand_rot = rotate_x(current_quat = [0.707, 0, 0.707, 0],angle_deg=self.random_degree)

        x_micwave = random.uniform(0.95, 1.0)
        y_micwave = random.uniform(-0.6,0.6)
        # x_micwave = 0.85
        # y_micwave = 0.9
        
        self.scene_key = instance_name
        micwave_cfg = AssetCfg(
            fix_base=False,
            name=self.scene_key,
            urdf_path=urdf_path,
            pos=(x_micwave, y_micwave, 0.45),
            rot=rot_real,
            scale=self.scale_micwave,
            asset_kind = "articulation",
            collider_type="convex_decomposition",
            stiffness=0.0,
            damping=0.0,
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
            compliant_contact_damping=0.0,
            compliant_contact_stiffness=0.0,
        )
        # 3) concept urdf 
        resolved_concept_urdf, model_id = resolve_concept_urdf_path(
            source_urdf_path=micwave_cfg.urdf_path,
            concept_root=concept_root,
            category=category,
        )
        if resolved_concept_urdf is None:
            print(f"[WARN] model_id={model_id} not found, fallback: {concept_urdf_fallback}")
            resolved_concept_urdf = concept_urdf_fallback
        else:
            print(f"[INFO] model_id={model_id}, concept URDF: {resolved_concept_urdf}")
        #concept urdf will be used in the atomic task
        self.concept_urdf_path = resolved_concept_urdf
        
        concept_micwave_cfg = AssetCfg(
            name="my_concept_micwave",
            urdf_path=resolved_concept_urdf,
            pos=(0.9, 0, 0.35),
            rot=rot_concept,
            scale=self.scale_micwave,
            asset_kind = "articulation",
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
            compliant_contact_damping=0.0,
            compliant_contact_stiffness=0.0,
        )
        return [micwave_cfg]
    def open(self,scene,is_clockwise:None,env_id):

        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}" 
            microwave_asset = scene[self.actual_instance_name]
            my_concept_micwave = scene[f"my_concept_micwave__env{envid}"]
            real_root_pos = microwave_asset.data.root_pos_w[0].detach().cpu().numpy()
            concept_root_pos = my_concept_micwave.data.root_pos_w[0].detach().cpu().numpy()

            h_pos, h_quat = get_handle_pose_physics(
                my_concept_micwave,
                self.scale_micwave,
                concept_urdf_path=self.concept_urdf_path,
            )
            h_pos_real = map_concept_point_to_real_xy(h_pos, concept_root_pos, real_root_pos)
            finger_length = 0.05
            print(f"the handle pose:{h_pos_real}")
            x1 = h_pos_real[0] 
            y1 = h_pos_real[1]
            z1 = h_pos_real[2]
            print(f"a important thingsaddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd{x1}dsgfddddddddddsg{y1}")

            hinge_origin_local, hinge_axis_local, hinge_parent, hinge_door = get_door_hinge_from_urdf(self.concept_urdf_path)
            hinge_origin_local = np.asarray(hinge_origin_local, dtype=np.float64) * np.asarray(self.scale_micwave, dtype=np.float64)
            body_names = list(my_concept_micwave.body_names)
            if hinge_parent in body_names:
                parent_ids, _ = my_concept_micwave.find_bodies(hinge_parent)
                pstate = my_concept_micwave.data.body_state_w[0, int(parent_ids[0])]
                parent_pos_w = pstate[0:3].detach().cpu().numpy()
                parent_quat_w = pstate[3:7].detach().cpu().numpy()
            else:
                # parent may be "world"/"base": use concept root pose
                parent_pos_w = concept_root_pos
                parent_quat_w = my_concept_micwave.data.root_quat_w[0].detach().cpu().numpy()

            _, axis_dir_concept = transform_joint_axis_to_world(
                parent_pos_w, parent_quat_w, hinge_origin_local, hinge_axis_local
            )
            # Door-link origin is the hinge. Step back from the handle by the URDF offset
            # so the pivot is not the parent body origin (that shortens the radius).
            offset, _, _ = get_door_to_handle_offset_from_urdf(self.concept_urdf_path)
            if offset is not None:
                offset_s = np.asarray(offset, dtype=np.float64) * np.asarray(self.scale_micwave, dtype=np.float64)
                qw, qx, qy, qz = h_quat
                axis_point_concept = np.asarray(h_pos, dtype=np.float64) - R.from_quat([qx, qy, qz, qw]).apply(offset_s)
            else:
                axis_point_concept = transform_joint_axis_to_world(
                    parent_pos_w, parent_quat_w, hinge_origin_local, hinge_axis_local
                )[0]
            axis_point_real = map_concept_point_to_real_xy(axis_point_concept, concept_root_pos, real_root_pos)
            axis_dir_real = np.asarray(axis_dir_concept, dtype=np.float64)
            if is_clockwise:
                axis_dir_real = -axis_dir_real
            handle_xyz_dbg = np.array([x1, y1, z1], dtype=np.float64)
            radius = float(np.linalg.norm(np.cross(axis_dir_real, handle_xyz_dbg - axis_point_real)))
            print(f"hinge matched: parent={hinge_parent}, door={hinge_door}, radius={radius:.3f}")

            hand_to_world_rot_wxyz = rotate_x([0.707, 0, 0.707, 0], -random_degree)
            r_hand_world = R.from_quat(list(wxyz_to_xyzw(hand_to_world_rot_wxyz)))
            handle_xyz = np.array([x1, y1, z1], dtype=np.float64)
            # Grasp 0.08 m further along +hand Z. Do not back the palm off by the finger length.
            grasp_xyz = handle_xyz + r_hand_world.apply(np.array([0.0, 0.0, 0.06], dtype=np.float64))
            arc_points = generate_handle_arc_points_about_axis(
                axis_point=axis_point_real,
                axis_dir=axis_dir_real,
                handle_pos0=grasp_xyz,
                max_deg=90,
                step_deg=20,
            )
            # After 40 deg, pull each point 10% of the way toward the hinge.
            _degrees = list(range(0, 70 + 1, 20))
            if _degrees[-1] != 70:
                _degrees.append(70)
            _axis = np.asarray(axis_dir_real, dtype=np.float64)
            _axis = _axis / max(float(np.linalg.norm(_axis)), 1e-8)
            _pivot = np.asarray(axis_point_real, dtype=np.float64)
            _shrunk = []
            for _deg, _p in zip(_degrees, arc_points):
                _p = np.asarray(_p, dtype=np.float64)
                if _deg > 40:
                    _radial = _p - _pivot
                    _radial = _radial - np.dot(_radial, _axis) * _axis
                    _p = _p - _radial * 0.19
                _shrunk.append([float(_p[0]), float(_p[1]), float(_p[2])])
            arc_points = _shrunk
            print(f"[microwave] open arc after 40 deg uses 90% of radius={radius:.3f}")
            # arc_goal_poses = [
            #     [p[0], p[1], p[2], 0.707, 0, 0.707, 0]
            #     for p in arc_points
            # ]
            # goal_pose_grasp_pre = [x1-0.45, y1-0.1 , z1 ,0.707, 0, 0.707, 0]
            # goal_pose_grasp = [x1-0.15, y1-0.1, z1 ,0.707, 0, 0.707, 0]
            # last_pose = arc_goal_poses[-1].copy()  # [x,y,z,qw,qx,qy,qz]
            # last_pose[0] -= 0.1
            # away_pose = last_pose
            # away_pose[2] += 0.3
            # last_pose[1] += 0.3

            pos_pre = handle_xyz + r_hand_world.apply(np.array([0.0, 0.0, -0.1], dtype=np.float64))
            pos_pre2 = grasp_xyz
            hand_to_world_rot_wxyz = xyzw_to_wxyz(r_hand_world.as_quat())
            self.open_hand_quat = hand_to_world_rot_wxyz
            dz = np.random.uniform(-0.03,0.03)

            pose1 = [pos_pre[0], pos_pre[1],pos_pre[2] + dz, hand_to_world_rot_wxyz[0], hand_to_world_rot_wxyz[1], hand_to_world_rot_wxyz[2],hand_to_world_rot_wxyz[3]]
            pose2 = [pos_pre2[0], pos_pre2[1],pos_pre2[2] + dz, hand_to_world_rot_wxyz[0], hand_to_world_rot_wxyz[1], hand_to_world_rot_wxyz[2],hand_to_world_rot_wxyz[3]]
            # pose3 = [pos_pre3[0], pos_pre3[1],pos_pre3[2], hand_to_world_rot_wxyz[0], hand_to_world_rot_wxyz[1], hand_to_world_rot_wxyz[2],hand_to_world_rot_wxyz[3]]
            arc_goal_poses = [
                [p[0], p[1], p[2] + dz ,  hand_to_world_rot_wxyz[0], hand_to_world_rot_wxyz[1], hand_to_world_rot_wxyz[2],hand_to_world_rot_wxyz[3]]
                for p in arc_points
            ]
            # pose3 = [x1,0.35,0.45,0.707, 0, 0.707, 0]
            # pose4 = [x1,0.45,0.45,0.707, 0, 0.707, 0]
            # pose5 = [0.30,0.5,0.45,0.707, 0, 0.707, 0]
            # pose6 = [0.25,0.5,0.45,0.707, 0, 0.707, 0]
            if y1 >0:
                executor = "robot_left"
            else:
                executor = "robot_right"          

            pose_pack = [
                {"pose": pose1, "gripper": "open","use_obstacle_avoidance": False,"executor":executor},
                {"pose": pose2,     "gripper": "open","use_obstacle_avoidance": False,"executor":executor},
                # {"pose": pose3,     "gripper": "close","use_obstacle_avoidance": False,"executor":"robot_left"},
                *[{"pose": p, "gripper": "close","use_obstacle_avoidance": False,"executor":executor} for p in arc_goal_poses],
                # {"pose":pose3,"gripper":"close","use_obstacle_avoidance": True,"executor":"robot_left"},
                # {"pose":pose4,"gripper":"close","use_obstacle_avoidance": True,"executor":"robot_left"},
                # {"pose":pose5,"gripper":"close","use_obstacle_avoidance": True,"executor":"robot_left"},
                # {"pose":pose6,"gripper":"close","use_obstacle_avoidance": True,"executor":"robot_left"}
            ]
            for wp in pose_pack:
                wp["ori_tol_deg"] = 80.0
            return pose_pack

    def open_release(self, scene, is_clockwise=None, env_id=0):
        """Open the door, then let go and back up so the same hand can pick."""
        pack = self.open(scene, is_clockwise, env_id)
        if not pack:
            return pack
        last = pack[-1]
        pose = list(last["pose"])
        released = dict(last)
        released["gripper"] = "open"
        retreat = dict(last)
        retreat_pose = list(pose)
        retreat_pose[0] = float(retreat_pose[0]) - 0.15
        retreat["pose"] = retreat_pose
        retreat["gripper"] = "open"
        return list(pack) + [released, retreat]

    def close(self, scene, is_clockwise=None, env_id=0):
        """Close along the open hinge, starting from the live door joint and root pose."""
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            microwave_asset = scene[self.actual_instance_name]
            my_concept_micwave = scene[f"my_concept_micwave__env{envid}"]
            real_root_pos = microwave_asset.data.root_pos_w[0].detach().cpu().numpy()
            real_root_quat = microwave_asset.data.root_quat_w[0].detach().cpu().numpy()
            concept_root_pos = my_concept_micwave.data.root_pos_w[0].detach().cpu().numpy()

            real_jname, q_real = largest_abs_joint(microwave_asset, 0)
            close_deg = max(10, int(round(abs(np.degrees(q_real)))))
            print(f"[microwave] close from live joint {real_jname}={q_real:.3f} deg={close_deg}")

            h_pos, h_quat = get_handle_pose_physics(
                my_concept_micwave,
                self.scale_micwave,
                concept_urdf_path=self.concept_urdf_path,
            )
            h_pos_real = map_concept_point_to_real_xy(h_pos, concept_root_pos, real_root_pos)

            hinge_origin_local, hinge_axis_local, hinge_parent, hinge_door = get_door_hinge_from_urdf(self.concept_urdf_path)
            hinge_origin_local = np.asarray(hinge_origin_local, dtype=np.float64) * np.asarray(self.scale_micwave, dtype=np.float64)
            body_names = list(my_concept_micwave.body_names)
            if hinge_parent in body_names:
                parent_ids, _ = my_concept_micwave.find_bodies(hinge_parent)
                pstate = my_concept_micwave.data.body_state_w[0, int(parent_ids[0])]
                parent_pos_w = pstate[0:3].detach().cpu().numpy()
                parent_quat_w = pstate[3:7].detach().cpu().numpy()
            else:
                parent_pos_w = concept_root_pos
                parent_quat_w = my_concept_micwave.data.root_quat_w[0].detach().cpu().numpy()

            _, axis_dir_concept = transform_joint_axis_to_world(
                parent_pos_w, parent_quat_w, hinge_origin_local, hinge_axis_local
            )
            offset, _, _ = get_door_to_handle_offset_from_urdf(self.concept_urdf_path)
            if offset is not None:
                offset_s = np.asarray(offset, dtype=np.float64) * np.asarray(self.scale_micwave, dtype=np.float64)
                qw, qx, qy, qz = h_quat
                axis_point_concept = np.asarray(h_pos, dtype=np.float64) - R.from_quat([qx, qy, qz, qw]).apply(offset_s)
            else:
                axis_point_concept = transform_joint_axis_to_world(
                    parent_pos_w, parent_quat_w, hinge_origin_local, hinge_axis_local
                )[0]
            axis_point_real = map_concept_point_to_real_xy(axis_point_concept, concept_root_pos, real_root_pos)
            # Same axis as open. The arc is built from the closed handle, then reversed
            # so it starts at the live open angle and swings shut.
            axis_dir_real = np.asarray(axis_dir_concept, dtype=np.float64)
            if is_clockwise:
                axis_dir_real = -axis_dir_real

            hand_to_world_rot_wxyz = rotate_x([0.707, 0, 0.707, 0], -random_degree)
            r_hand_world = R.from_quat(list(wxyz_to_xyzw(hand_to_world_rot_wxyz)))
            grasp_xyz = np.asarray(h_pos_real, dtype=np.float64) + r_hand_world.apply(
                np.array([0.0, 0.0, 0.08], dtype=np.float64)
            )
            axis_n = np.asarray(axis_dir_real, dtype=np.float64)
            axis_n = axis_n / max(float(np.linalg.norm(axis_n)), 1e-8)
            radial = grasp_xyz - np.asarray(axis_point_real, dtype=np.float64)
            radial = radial - np.dot(radial, axis_n) * axis_n
            radius = float(np.linalg.norm(radial))
            shrink = 0.04
            if radius > shrink + 1e-4:
                grasp_xyz = grasp_xyz - radial * (shrink / radius)
            print(
                f"[microwave] close grasp at live {close_deg} deg, "
                f"radius {radius:.3f}->{max(radius - shrink, 0.0):.3f}"
            )
            arc_points = list(reversed(generate_handle_arc_points_about_axis(
                axis_point=axis_point_real,
                axis_dir=axis_dir_real,
                handle_pos0=grasp_xyz,
                max_deg=close_deg,
                step_deg=10,
            )))
            q = xyzw_to_wxyz(r_hand_world.as_quat())
            shut_poses = [
                [p[0], p[1], p[2], q[0], q[1], q[2], q[3]]
                for p in arc_points
            ]
            shut_poses[0][1] = float(shut_poses[0][1]) + 0.05
            retreat = [shut_poses[-1][0], shut_poses[-1][1], shut_poses[-1][2] + 0.12, q[0], q[1], q[2], q[3]]
            print(f"[microwave] close door hinge parent={hinge_parent} door={hinge_door} pts={len(shut_poses)}")
            pack = [
                {"pose": shut_poses[0], "gripper": "close", "use_obstacle_avoidance": False, "executor": "robot_left"},
                *[{"pose": p, "gripper": "close", "use_obstacle_avoidance": False, "executor": "robot_left"} for p in shut_poses[1:]],
                {"pose": shut_poses[-1], "gripper": "open", "use_obstacle_avoidance": False, "executor": "robot_left"},
                {"pose": retreat, "gripper": "open", "use_obstacle_avoidance": False, "executor": "robot_left"},
            ]
            return repose_waypoints(pack, real_root_pos, real_root_quat, rot_real, 0.42)

    def right(self,scene, env_id=0):
        
        right_range=(0.55, 0.85)    # 右侧距离范围（米）
        forward_range=(-0.15, 0.15)  # 前后微调
        height_range=(-0.15, 0.05)
        
        microwave_asset = scene["my_micwave"]
        root_pos = microwave_asset.data.root_pos_w[0]      

        x_microwave = root_pos[0].item()
        y_microwave = root_pos[1].item()
        z_microwave = root_pos[2].item()

        # 约定：局部 +x 前，+y 左，所以“右侧”是 -y
        dx = np.random.uniform(*forward_range)
        dy = -np.random.uniform(*right_range)
        dz = np.random.uniform(*height_range)


        place_pose = [x_microwave + dx,y_microwave + dy,z_microwave + dz,self.hand_rot[0], self.hand_rot[1], self.hand_rot[2], self.hand_rot[3]]

        goal_pose = [
            {"pose":place_pose,"gripper":"close","use_obstacle_avoidance": True}
        ]
    
        return goal_pose
        
    def on(self,scene):
        
        right_range=(0.1, 0.1)    # 右侧距离范围（米）
        forward_range=(-0.15, 0.15)  # 前后微调

        
        microwave_asset = scene["my_micwave"]
        root_pos = microwave_asset.data.root_pos_w[0]      

        x_microwave = root_pos[0].item()
        y_microwave = root_pos[1].item()
        z_microwave = root_pos[2].item()

        # 约定：局部 +x 前，+y 左，所以“右侧”是 -y
        dx = np.random.uniform(*forward_range)
        dy = -np.random.uniform(*right_range)


        self.hand_rot = rotate_x(current_quat = [0.707, 0, 0.707, 0],angle_deg=random_degree)
        place_pose = [x_microwave + dx,y_microwave + dy,z_microwave + 0.4,self.hand_rot[0], self.hand_rot[1], self.hand_rot[2], self.hand_rot[3]]
        after_place_pose = [x_microwave-0.2 + dx,y_microwave + dy,z_microwave + 0.3,self.hand_rot[0], self.hand_rot[1], self.hand_rot[2], self.hand_rot[3]]

        goal_pose = [
            {"pose":place_pose,"gripper":"close","use_obstacle_avoidance": True},
            {"pose":after_place_pose,"gripper":"open","use_obstacle_avoidance": True}
        ]
    
        return goal_pose
    
    def into(self, scene, env_id=0):
        right_range=(0.05, 0.05)    # 右侧距离范围（米）
        forward_range=(-0.05, 0.05)  # 前后微调

        name = f"{self.scene_key}__env{env_id}"
        try:
            microwave_asset = scene[name]
        except (KeyError, AttributeError):
            microwave_asset = scene[self.scene_key]
        root_pos = microwave_asset.data.root_pos_w[0]

        x_microwave = root_pos[0].item()
        y_microwave = root_pos[1].item()
        z_microwave = root_pos[2].item()

        # 约定：局部 +x 前，+y 左，所以“右侧”是 -y
        dx = np.random.uniform(*forward_range)
        dy = -np.random.uniform(*right_range)

        self.hand_rot = rotate_x(current_quat=[0.707, 0, 0.707, 0], angle_deg=random_degree)
        q = self.hand_rot
        hand_rot = R.from_quat(list(wxyz_to_xyzw(q)))
        y_shift = hand_rot.apply(np.array([0.0, -0.14, 0.0]))
        place_shift = hand_rot.apply(np.array([0.0, -0.14, -0.10]))
        pre_place_pose = [x_microwave - 0.4 + y_shift[0], y_microwave + y_shift[1], z_microwave + 0.04 + y_shift[2], q[0], q[1], q[2], q[3]]
        place_pose = [x_microwave + y_shift[0], y_microwave + y_shift[1], z_microwave + 0.04 + y_shift[2], q[0], q[1], q[2], q[3]]
        after_place_pose = [x_microwave - 0.4 + dx + y_shift[0], y_microwave  + y_shift[1], z_microwave + 0.04 + y_shift[2], q[0], q[1], q[2], q[3]]
        # Bread sits on the negative-y side, so the right arm is holding it.
        # Pre-place, place, and retreat all avoid the live microwave and the table.
        goal_pose = [
            {
                "pose": pre_place_pose,
                "gripper": "close",
                "use_obstacle_avoidance": False,
                "avoid_table": False,
                "avoid_microwave": False,
                "executor": "robot_right",
            },
            {
                "pose": place_pose,
                "gripper": "close",
                "use_obstacle_avoidance": False,
                "avoid_table": False,
                "avoid_microwave": False,
                "executor": "robot_right",
            },
            {
                "pose": after_place_pose,
                "gripper": "open",
                "use_obstacle_avoidance": False,
                "avoid_table": False,
                "avoid_microwave": False,
                "executor": "robot_right",
            },
        ]
        return goal_pose

    def into_left(self, scene, env_id=0):
        """Place a left-arm grasp into the microwave, then back out."""
        name = f"{self.scene_key}__env{env_id}"
        try:
            microwave_asset = scene[name]
        except (KeyError, AttributeError):
            microwave_asset = scene[self.scene_key]
        root_pos = microwave_asset.data.root_pos_w[0]
        x_microwave = root_pos[0].item()
        y_microwave = root_pos[1].item()
        z_microwave = root_pos[2].item()
        q = getattr(self, "open_hand_quat", None) or rotate_x([0.707, 0, 0.707, 0], -random_degree)
        self.hand_rot = q
        hand_rot = R.from_quat(list(wxyz_to_xyzw(q)))
        y_shift = hand_rot.apply(np.array([0.0, -0.06, 0.0]))
        y_shift2 = hand_rot.apply(np.array([0.0, -0.01, 0.0]))
        place_x = x_microwave - 0.1
        approach_x = x_microwave - 0.55
        place_y = y_microwave + y_shift2[1]
        place_z = z_microwave + 0.04 + y_shift[2]
        q_list = [q[0], q[1], q[2], q[3]]
        step = 0.03
        n_seg = max(1, int(round(abs(place_x - approach_x) / step)))
        drop_poses = [
            [float(x), place_y, place_z, *q_list]
            for x in np.linspace(approach_x, place_x, n_seg + 1)
        ]
        after_place_pose = [approach_x, place_y, place_z, *q_list]
        print(
            f"[microwave] into_left {n_seg} segments of {step:.2f} m along x "
            f"yz=({place_y:.3f},{place_z:.3f}) x={approach_x:.3f}->{place_x:.3f}"
        )
        return [
            *[
                {
                    "pose": pose,
                    "gripper": "close",
                    "use_obstacle_avoidance": False,
                    "avoid_table": False,
                    "avoid_microwave": False,
                    "executor": "robot_left",
                }
                for pose in drop_poses
            ],
            {
                "pose": after_place_pose,
                "gripper": "open",
                "use_obstacle_avoidance": False,
                "avoid_table": False,
                "avoid_microwave": False,
                "executor": "robot_left",
            },
        ]