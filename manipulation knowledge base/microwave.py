from my_utils import *
from set_interface_try import *
import random

rot_real, random_degree = rotate_z(
            current_quat=[1, 0, 0, 0],
            angle_deg=float(random.uniform(40, 50.0)),
        )
class mircrowave ():
    def __init__(self,):
        self.scale_micwave = (0.2,0.2,0.2)
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

        x_micwave = random.uniform(0.8, 0.9)
        y_micwave = random.uniform(0.1, 0.2)
        x_micwave = 0.85
        y_micwave = 0.6
        
        self.scene_key = instance_name
        micwave_cfg = AssetCfg(
            fix_base=False,
            name=self.scene_key,
            urdf_path=urdf_path,
            pos=(x_micwave, y_micwave, 0.45),
            rot=rot_real,
            scale=self.scale_micwave,
            asset_kind = "articulation",
            # collider_type="convex_decomposition",
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
            pos=(0.9, 0, 0.3),
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

            axis_point_concept, axis_dir_concept = transform_joint_axis_to_world(
                parent_pos_w, parent_quat_w, hinge_origin_local, hinge_axis_local
            )
            axis_point_real = map_concept_point_to_real_xy(axis_point_concept, concept_root_pos, real_root_pos)
            axis_dir_real = np.asarray(axis_dir_concept, dtype=np.float64)
            if is_clockwise:
                axis_dir_real = -axis_dir_real
            print(f"hinge matched: parent={hinge_parent}, door={hinge_door}")

            arc_points = generate_handle_arc_points_about_axis(
            axis_point=axis_point_real,
            axis_dir=axis_dir_real,
            handle_pos0=np.array([x1, y1, z1], dtype=np.float64),
            max_deg=25,
            step_deg=5,
            )
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

            pos_w = [x1,y1,z1]

            # r_x = R.from_euler('x', 90, degrees=True)
            # r_y = R.from_euler('y',90, degrees=True)
            # r_hand_obj = r_y*r_x    # 先z后x
            # # self.rot1 若是 wxyz，要先转
            # r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            # r_hand_world = r_obj_world * r_hand_obj
            # q_hand_world_xyzw = r_hand_world.as_quat()
            hand_to_world_rot_wxyz = rotate_x([0.707, 0, 0.707, 0], -random_degree)
            r_hand_world = R.from_quat(list(wxyz_to_xyzw(hand_to_world_rot_wxyz)))
            pos_w = np.asarray(pos_w, dtype=np.float64)
            pos_pre = pos_w - r_hand_world.apply(np.array([0.0, 0, 0.1], dtype=np.float64))
            pos_pre2= pos_w - r_hand_world.apply(np.array([0.0, 0, -0.05], dtype=np.float64))
            pos_pre3= pos_w - r_hand_world.apply(np.array([0.0, 0, 0.2], dtype=np.float64))
            hand_to_world_rot_wxyz = xyzw_to_wxyz(r_hand_world.as_quat())

            pose1 = [pos_pre[0], pos_pre[1],pos_pre[2], hand_to_world_rot_wxyz[0], hand_to_world_rot_wxyz[1], hand_to_world_rot_wxyz[2],hand_to_world_rot_wxyz[3]]
            pose2 = [pos_pre2[0], pos_pre2[1],pos_pre2[2], hand_to_world_rot_wxyz[0], hand_to_world_rot_wxyz[1], hand_to_world_rot_wxyz[2],hand_to_world_rot_wxyz[3]]
            # pose3 = [pos_pre3[0], pos_pre3[1],pos_pre3[2], hand_to_world_rot_wxyz[0], hand_to_world_rot_wxyz[1], hand_to_world_rot_wxyz[2],hand_to_world_rot_wxyz[3]]
            arc_goal_poses = [
                [p[0], p[1], p[2],  hand_to_world_rot_wxyz[0], hand_to_world_rot_wxyz[1], hand_to_world_rot_wxyz[2],hand_to_world_rot_wxyz[3]]
                for p in arc_points
            ]
            # pose3 = [x1,0.35,0.45,0.707, 0, 0.707, 0]
            # pose4 = [x1,0.45,0.45,0.707, 0, 0.707, 0]
            # pose5 = [0.30,0.5,0.45,0.707, 0, 0.707, 0]
            # pose6 = [0.25,0.5,0.45,0.707, 0, 0.707, 0]
            
    
            pose_pack = [
                {"pose": pose1, "gripper": "open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose": pose2,     "gripper": "open","use_obstacle_avoidance": False,"executor":"robot_left"},
                # {"pose": pose3,     "gripper": "close","use_obstacle_avoidance": False,"executor":"robot_left"},
                *[{"pose": p, "gripper": "close","use_obstacle_avoidance": False,"executor":"robot_left"} for p in arc_goal_poses],
                # {"pose":pose3,"gripper":"close","use_obstacle_avoidance": True,"executor":"robot_left"},
                # {"pose":pose4,"gripper":"close","use_obstacle_avoidance": True,"executor":"robot_left"},
                # {"pose":pose5,"gripper":"close","use_obstacle_avoidance": True,"executor":"robot_left"},
                # {"pose":pose6,"gripper":"close","use_obstacle_avoidance": True,"executor":"robot_left"}
            ]        
            return pose_pack
    
    def right(self,scene):
        
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


        self.hand_rot,_ = concept_rotate(current_quat = [0.707, 0, 0.707, 0],angle_deg=random_degree)
        place_pose = [x_microwave + dx,y_microwave + dy,z_microwave + 0.4,self.hand_rot[0], self.hand_rot[1], self.hand_rot[2], self.hand_rot[3]]
        after_place_pose = [x_microwave-0.2 + dx,y_microwave + dy,z_microwave + 0.3,self.hand_rot[0], self.hand_rot[1], self.hand_rot[2], self.hand_rot[3]]

        goal_pose = [
            {"pose":place_pose,"gripper":"close","use_obstacle_avoidance": True},
            {"pose":after_place_pose,"gripper":"open","use_obstacle_avoidance": True}
        ]
    
        return goal_pose
    
    def into(self,scene):
        right_range=(0.05, 0.05)    # 右侧距离范围（米）
        forward_range=(-0.05, 0.05)  # 前后微调

        
        microwave_asset = scene["my_micwave"]
        root_pos = microwave_asset.data.root_pos_w[0]      

        x_microwave = root_pos[0].item()
        y_microwave = root_pos[1].item()
        z_microwave = root_pos[2].item()

        # 约定：局部 +x 前，+y 左，所以“右侧”是 -y
        dx = np.random.uniform(*forward_range)
        dy = -np.random.uniform(*right_range)


        self.hand_rot,_ = concept_rotate(current_quat = [0.707, 0, 0.707, 0],angle_deg=self.random_degree)
        pre_place_pose = [x_microwave -0.4,y_microwave + dy,z_microwave,self.hand_rot[0], self.hand_rot[1], self.hand_rot[2], self.hand_rot[3]]
        place_pose = [x_microwave ,y_microwave ,z_microwave,self.hand_rot[0], self.hand_rot[1], self.hand_rot[2], self.hand_rot[3]]
        after_place_pose = [x_microwave-0.4 + dx,y_microwave + dy,z_microwave ,self.hand_rot[0], self.hand_rot[1], self.hand_rot[2], self.hand_rot[3]]

        goal_pose = [
            {"pose":pre_place_pose,"gripper":"close","use_obstacle_avoidance": True},
            {"pose":place_pose,"gripper":"close","use_obstacle_avoidance": False},
            {"pose":after_place_pose,"gripper":"open","use_obstacle_avoidance": False},
        ]
    
        return goal_pose