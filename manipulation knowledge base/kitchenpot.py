from my_utils import *
from set_interface_try import *
import random
from kitchenpot_concept import random_degree
ZONES = [
    ((0.5, 0.7), (-0.1, 0.1)),  # zone 0

]
target_pos_x = np.random.uniform(0.4,0.5)
target_pos_y = np.random.uniform(-0.2,0.2)
target_pos_z = 0.3 



def sample_pos_by_instance(instance_name: str):
    # 从名字里取最后数字，如 my_bowl_bowl_3 -> 3
    m = re.search(r"(?:^|__)env(\d+)$", instance_name)
    if m:
        idx = int(m.group(1))
    else:
        m2 = re.search(r"_(\d+)$", instance_name)  # 兼容旧格式 xxx_1
        if not m2:
            raise ValueError(f"Cannot parse index from instance_name={instance_name}")
        idx = int(m2.group(1)) - 1
    xr, yr = ZONES[idx % len(ZONES)]
    x = random.uniform(*xr)
    y = random.uniform(*yr)
    return x, y
class kitchenpot ():
    def __init__(self):
        self.scale = (0.18,0.18,0.18)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs:int
    ):
        self.num_envs = num_envs
        self.angle_degree = random_degree
        print(f"the value of angle_degree{self.angle_degree} ")
        x1, y1 = sample_pos_by_instance(instance_name)  # 或你自己的分区函数
        r_z = R.from_euler("z", self.angle_degree, degrees=True)
        r_x = R.from_euler("x", 0, degrees=True) 
        self.q_obj_world_wxyz = xyzw_to_wxyz((r_z * r_x).as_quat())
        concept_root = str(PARTNET_CONCEPT_ROOT)
        category = "KitchenPot"
        concept_urdf_fallback= str(PARTNET_CONCEPT_ROOT / "kitchenpotPot" / "1fde7ce0a20f1502f54e1b6f41fdd78a" / "KitchenPot.urdf")


        rot_real, self.random_degree = random_rotate_mic(current_quat=[1, 0, 0, 0], random_degree_range=0.0)
        rot_concept = rotate_x(current_quat=[0.707, 0, 0, -0.707], angle_deg=self.random_degree)
        # self.hand_rot = rotate_x(current_quat = [0.707, 0, 0.707, 0],angle_deg=self.random_degree)

        
        self.scene_key = instance_name
        cfg = AssetCfg(
            fix_base=False,
            name=self.scene_key,
            urdf_path=urdf_path,
            pos=(x1, y1, 0.45),
            rot=self.q_obj_world_wxyz,
            scale=self.scale,
            asset_kind = "articulation",
            stiffness=0.0,
            damping=1.0,
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
            compliant_contact_damping=0.0,
            compliant_contact_stiffness=0.0,
            # joint_pos = {"joint_1": -0.8},
        )
        # 3) concept urdf 
        resolved_concept_urdf, model_id = resolve_concept_urdf_path(
            source_urdf_path=cfg.urdf_path,
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
        

        return [cfg]


    def pick_simultaneously_left(self,scene,env_id):
    
        for envid in range(self.num_envs):
            
            self.actual_instance_name = f"{self.scene_key}__env{envid}" 
            microwave_asset = scene[self.actual_instance_name]
            my_concept = scene[f"my_concept_kitchenpot__env{envid}"]
            real_root_pos = microwave_asset.data.root_pos_w[0].detach().cpu().numpy()
            concept_root_pos = my_concept.data.root_pos_w[0].detach().cpu().numpy()
            
            pos, quat, matched_name = get_first_available_handle_pose(
            asset=my_concept,
            urdf_path=self.concept_urdf_path,
            scale=self.scale,
            env_id=envid,
            candidates=["L_Shaped_Sidehandle", "Trifold_Sidehandle"],
        )
            h_pos_real = map_concept_point_to_real_xy(pos, concept_root_pos, real_root_pos)
            finger_length = 0.05
            print(f"the handle pose:{h_pos_real}")
            x1 = h_pos_real[0] - finger_length
            y1 = h_pos_real[1]
            z1 = h_pos_real[2]
            print(f"the pose of the screen{h_pos_real}fjhhfffffffffffffffffffffffffffffffffffff")


            self.actual_instance_name = f"{self.scene_key}__env{envid}"    
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
     
    
    
            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
           
    
            target_pos_x = x1 
            target_pos_y = y1 
            target_pos_z = z1
    

            r_x = R.from_euler('x', 90, degrees=True)  #世界坐标z轴
            r_z = R.from_euler('z',0, degrees=True)  #夹爪的旋转
            # r_z = R.from_euler('z',90, degrees=True)
            r_hand_obj = r_z*r_x    # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            pos_w = [x1,y1,z1]
            pos1 = pos_w - hand_to_world_rot.apply([-0.05,0.0,0.4])
            pos2 = pos_w - hand_to_world_rot.apply([-0.05,0.0,0.22])
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
            
    
    
            pose1= [pos1[0],pos1[1],target_pos_z-0.22,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [pos2[0],pos2[1],target_pos_z-0.22,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [pos2[0],pos2[1],target_pos_z,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [pos2[0],pos2[1],target_pos_z,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose5 = [pos2[0],pos2[1],target_pos_z,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
        
    
    
            if y >0:
                executor = "robot_left"
            else:
                executor = "robot_right"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left","sync_id": "pick_1"},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
    
            ]
            return pose
    def pick_simultaneously_right(self,scene,env_id):
    
        for envid in range(self.num_envs):


            self.actual_instance_name = f"{self.scene_key}__env{envid}" 
            microwave_asset = scene[self.actual_instance_name]
            my_concept = scene[f"my_concept_kitchenpot__env{envid}"]
            real_root_pos = microwave_asset.data.root_pos_w[0].detach().cpu().numpy()
            concept_root_pos = my_concept.data.root_pos_w[0].detach().cpu().numpy()
            
            pos, quat, matched_name = get_first_available_handle_pose(
            asset=my_concept,
            urdf_path=self.concept_urdf_path,
            scale=self.scale,
            env_id=envid,
            candidates=["L_Shaped_Sidehandle", "Trifold_Sidehandle"],
        )
            h_pos_real = map_concept_point_to_real_xy(pos, concept_root_pos, real_root_pos)
            finger_length = 0.05
            print(f"the handle pose:{h_pos_real}")


            x1 = h_pos_real[0] - finger_length
            y1 = h_pos_real[1]
            z1 = h_pos_real[2]
            print(f"the pose of the screen{h_pos_real}fjhhfffffffffffffffffffffffffffffffffffff")
            self.actual_instance_name = f"{self.scene_key}__env{envid}"    
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
     
    
    
            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
           
    
            target_pos_x = x1
            target_pos_y = y1 
            target_pos_z = z1
    

            r_x = R.from_euler('x', -90, degrees=True)  #世界坐标z轴
            r_z = R.from_euler('z',0, degrees=True)  #夹爪的旋转
            # r_z = R.from_euler('z',90, degrees=True)
            r_hand_obj = r_z*r_x    # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            pos_w = [x1,y1,z1]
            pos1 = pos_w - hand_to_world_rot.apply([-0.05,0.0,0.4])
            pos2 = pos_w - hand_to_world_rot.apply([-0.05,0.0,0.22])
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)  
            
    
    
            pose1= [pos1[0],pos1[1],target_pos_z-0.22,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [pos2[0],pos2[1],target_pos_z-0.22,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [pos2[0],pos2[1],target_pos_z+0.1,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [pos2[0],pos2[1],target_pos_z+0.1,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose5 = [pos2[0],pos2[1],target_pos_z+0.1,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
        
    
    
            if y >0:
                executor = "robot_left"
            else:
                executor = "robot_right"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right","sync_id": "pick_1"},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                    
    
            ]
            return pose

    def pick_simultaneously_right(self,scene,env_id):
    
        for envid in range(self.num_envs):


            self.actual_instance_name = f"{self.scene_key}__env{envid}" 
            microwave_asset = scene[self.actual_instance_name]
            my_concept = scene[f"my_concept_kitchenpot__env{envid}"]
            real_root_pos = microwave_asset.data.root_pos_w[0].detach().cpu().numpy()
            concept_root_pos = my_concept.data.root_pos_w[0].detach().cpu().numpy()
            
            pos, quat, matched_name = get_first_available_handle_pose(
            asset=my_concept,
            urdf_path=self.concept_urdf_path,
            scale=self.scale,
            env_id=envid,
            candidates=["L_Shaped_Sidehandle", "Trifold_Sidehandle"],
        )
            h_pos_real = map_concept_point_to_real_xy(pos, concept_root_pos, real_root_pos)
            finger_length = 0.05
            print(f"the handle pose:{h_pos_real}")


            x1 = h_pos_real[0] - finger_length
            y1 = h_pos_real[1]
            z1 = h_pos_real[2]
            print(f"the pose of the screen{h_pos_real}fjhhfffffffffffffffffffffffffffffffffffff")
            self.actual_instance_name = f"{self.scene_key}__env{envid}"    
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
     
    
    
            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
           
    
            target_pos_x = x1
            target_pos_y = y1 
            target_pos_z = z1
    

            r_x = R.from_euler('x', -90, degrees=True)  #世界坐标z轴
            r_z = R.from_euler('z',0, degrees=True)  #夹爪的旋转
            # r_z = R.from_euler('z',90, degrees=True)
            r_hand_obj = r_z*r_x    # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            pos_w = [x1,y1,z1]
            pos1 = pos_w - hand_to_world_rot.apply([-0.05,0.0,0.4])
            pos2 = pos_w - hand_to_world_rot.apply([-0.05,0.0,0.22])
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)  
            
    
    
            pose1= [pos1[0],pos1[1],target_pos_z-0.22,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [pos2[0],pos2[1],target_pos_z-0.22,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [pos2[0],pos2[1],target_pos_z+0.1,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [pos2[0],pos2[1],target_pos_z+0.1,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose5 = [pos2[0],pos2[1],target_pos_z+0.1,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
        
    
    
            if y >0:
                executor = "robot_left"
            else:
                executor = "robot_right"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right","sync_id": "pick_1"},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
         
                    
    
            ]
            return pose


    def left(self,scene,env_id):
    
        for envid in range(self.num_envs):


            self.actual_instance_name = f"{self.scene_key}__env{envid}" 


            self.actual_instance_name = f"{self.scene_key}__env{envid}"    
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
     
    
    
            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
           
    

    

            q_hand_world_wxyz = rotate_y([0.707, 0, 0.707, 0],90)
            
    
    
            pose1= [x,y+0.2,z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [x,y+0.2,z+0.15,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [x,y+0.2,z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [x,y+0.2,z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
           
    
    

            executor = "robot_left"

            pose = [
                {"pose":pose1,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
               
                    
    
            ]
            return pose