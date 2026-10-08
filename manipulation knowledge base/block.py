from my_utils import *
from set_interface_try import *
import random


ZONES = [
    ((0.35, 0.70), (0.18, 0.45)),   # left
    ((0.35, 0.70), (-0.45, -0.18)),  # right
    ((0.40, 0.70), (-0.12, 0.12)),   # center
]
def sample_pos_by_instance(instance_name: str):
    # 从名字里取最后数字，如 my_block_block_3 -> 3
    idx = int(instance_name.split("_")[-1]) - 1
    xr, yr = ZONES[idx % len(ZONES)]
    x = random.uniform(*xr)
    y = random.uniform(*yr)
    return x, y
class block ():
    def __init__(self):
        self.scale = (0.05,0.05,0.05)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs
    ):
        
        
        # x1 = random.uniform(0.5, 0.8)
        # y1 = random.uniform(0.2,0.7)


        # #demo_dustbin_rubbish
        # x1 = 0.5
        # y1 = 0.35
        self.num_envs = num_envs

        x1, y1 = sample_pos_by_instance(instance_name)
        print(f"[RAND] spawn {instance_name} at xy=({x1:.3f}, {y1:.3f})")
                                                                        # (-60,60)
        self.rot,self.angle_degree= rotate_z(angle_deg=float(np.random.uniform(0,359)),current_quat=[1,0,0,0])

   
        r_z = R.from_euler("z", 0, degrees=True)
        r_x = R.from_euler("x", 90, degrees=True)
        self.q_obj_world_wxyz = xyzw_to_wxyz((r_z * r_x).as_quat())
        self.scene_key = instance_name
        size = np.random.uniform(0.01,0.06)
        cfg = AssetCfg(
            fix_base=False,
            name = instance_name,
            urdf_path=urdf_path,
            pos = (x1, y1, 0.4),
            rot = self.rot,
            scale = self.scale,
            stiffness=0.0,
            damping=0.0,
            source_kind = "cuboid",
            cuboid_size= (0.05,0.05,0.05),  #pad (0.1,0.07,0.0001)  #normal block (0.05,0.05,0.05)
            cuboid_color = (np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)),
            asset_kind = "rigid",
            # diffuse_color=(1.0,0.0,0.0),
            # roughness=0.5,
            # metallic=0.9,
            opacity=0.0,
            static_friction=0.8,
            dynamic_friction= 0.8,
            restitution= 0,
            compliant_contact_damping=0,
            compliant_contact_stiffness=0
        )

        return [cfg]
    
    def pick(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[0]      
    


            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
        

            height_range=(-0.05, 0.0)
            dz = np.random.uniform(*height_range)

            self.hand_rot,_ = hand_rotate([0.707, 0, 0.707, 0],-self.angle_degree)

            grasp_pose_pre= [x,y,z+0.2,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            grasp_pose = [x,y,z+0.05,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            adjust_pose = [x,y,z+0.1,0.707, 0, 0.707, 0]
            pose4 = [x,y,z+0.3,0.707, 0, 0.707, 0]
            pose5 = [x,y,z+0.3,0.707, 0, 0.707, 0]

            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":grasp_pose_pre,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":grasp_pose,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":adjust_pose,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
            ]
            return pose
    def beat(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
    


            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
        
            # dl = 0.05
            # x  -=  dl * np.sin(np.deg2rad(self.angel_degree))
            # y  -=  dl * np.cos(np.deg2rad(self.angel_degree))


            
            # self.hand_rot,_ = hand_rotate([0.707, 0, 0.707, 0],self.angle_degree)
            
            # hand_to_object = rotate_x([1,0,0,0],180)
            

            # object_to_world = self.rot
            # object_to_world = R.from_quat(object_to_world)
            # hand_to_object = R.from_quat(hand_to_object)
            # hand_to_world = object_to_world * hand_to_object
            # hand_to_world = list(hand_to_world.as_quat())
            # hand_to_world = xyzw_to_wxyz(hand_to_world)


            # hand_to_world_rot = R.from_quat(hand_to_world)
            # pos_w = [x,y,z]
            # pos_beat = pos_w - hand_to_world_rot.apply([0.05,0,0])


            r_z = R.from_euler('z', 90, degrees=True)
            r_x = R.from_euler('x', 180, degrees=True)
            r_hand_obj = r_x * r_z   # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.rot))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
            # apply 要用 Rotation 对象（xyzw）
            delta = r_hand_world.apply([-0.1, 0, 0])
            pos_beat = np.array([x, y, z], dtype=float) + delta

            # r_z = R.from_euler('z', 90, degrees=True)
            # r_x = R.from_euler('x',0, degrees=True)
            # r_hand_obj = r_x*r_z    # 先z后x
            # # self.rot1 若是 wxyz，要先转
            # r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            # r_hand_world = r_obj_world * r_hand_obj
            # q_hand_world_xyzw = r_hand_world.as_quat()
            # hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            # pos_w = [x,y,z]
            # pos_beat = pos_w - hand_to_world_rot.apply([0.2,0.0,0.0])
            # pos_contact = pos_w - hand_to_world_rot.apply([0.0,0,0.07])
            # q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)


            beat_pose= [pos_beat[0],pos_beat[1],pos_beat[2]+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            grasp_pose = [pos_beat[0],pos_beat[1],pos_beat[2]+0.06,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            adjust_pose = [pos_beat[0],pos_beat[1],pos_beat[2]+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]


            executor = "robot_left"

            pose = [
                {"pose":beat_pose,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":grasp_pose,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":grasp_pose,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                # {"pose":adjust_pose,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                # {"pose":adjust_pose,"gripper":"close","use_obstacle_avoidance": False,"executor":executor}
            ]
            return pose
    def hand_over(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]   
            
    
            print(f"the asset name :{self.actual_instance_name}the pose got form sim:{root_pos}")

            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
            print(f"the asset name :{self.actual_instance_name}the pose got form sim:{root_pos}")

            height_range=(-0.05, 0.0)
            dz = np.random.uniform(*height_range)
            
            self.hand_rot,_ = hand_rotate([0.707, 0, 0.707, 0],-self.angle_degree)
        

            handover_rot_left = rotate_x([0.707, 0, 0.707, 0],-90)
            handover_rot_right = rotate_x([0.707, 0, 0.707, 0],90)
            handover_rot_right,_ = rotate_z(list(handover_rot_right),90)
            x_range = (0,0.3)
            target_pos_x = 0.55
            target_pos_y = 0
            target_pos_z = 0.5

            pose1= [x,y,z+0.2,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose2 = [x,y,z+0.05,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose3 = [x,y,z+0.1,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose4 = [target_pos_x,0,0.5,handover_rot_left[0], handover_rot_left[1],handover_rot_left[2],handover_rot_left[3]]
            pose5 = [target_pos_x,-0.30,0.5,handover_rot_right[0], handover_rot_right[1],handover_rot_right[2],handover_rot_right[3]]
            pose6 = [target_pos_x,-0.10,0.5,handover_rot_right[0], handover_rot_right[1],handover_rot_right[2],handover_rot_right[3]]
            pose7 = [target_pos_x,0,0.5,handover_rot_left[0], handover_rot_left[1],handover_rot_left[2],handover_rot_left[3]]
            pose8 = [target_pos_x,-0.40,0.65,handover_rot_right[0], handover_rot_right[1],handover_rot_right[2],handover_rot_right[3]]
            pose9 = [target_pos_x,-0.30,z+0.05,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose10 = [target_pos_x,-0.30,0.65,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose11 = [target_pos_x,-0.30,0.65,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]

            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose5,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose7,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose8,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose9,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose10,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose11,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
            ]
            return pose


    def hand_over_left2right(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]   
            
    
            print(f"the asset name :{self.actual_instance_name}the pose got form sim:{root_pos}")

            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
            print(f"the asset name :{self.actual_instance_name}the pose got form sim:{root_pos}")

            height_range=(-0.05, 0.0)
            dz = np.random.uniform(*height_range)
            r_x = R.from_euler('x', 90, degrees=True)
            r_y = R.from_euler('y',90, degrees=True)
            r_hand_obj = r_y*r_x    # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            pos_w = [x,y,z]
            pos_pre = pos_w - hand_to_world_rot.apply([0.0,0,0.2])
            pos_contact = pos_w - hand_to_world_rot.apply([0.0,0,0.07])
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
            
            self.hand_rot = rotate_y([0.707, 0, 0.707, 0],90)
            

            handover_rot_left = rotate_x([0.707, 0, 0.707, 0],-90)
            handover_rot_right = rotate_x([0.707, 0, 0.707, 0],90)
            handover_rot_right,_ = rotate_z(list(handover_rot_right),90)
            x_range = (0,0.3)
            target_pos_x = 0.55
            target_pos_y = 0
            target_pos_z = 0.5

            pose1= [x,y,z+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [x,y,z+0.05,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [x,y,z+0.1,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [target_pos_x,0,0.5,handover_rot_right[0], handover_rot_right[1],handover_rot_right[2],handover_rot_right[3]]
            pose5 = [target_pos_x,-0.30,0.5,handover_rot_left[0], handover_rot_left[1],handover_rot_left[2],handover_rot_left[3]]
            pose6 = [target_pos_x,-0.10,0.5,handover_rot_left[0], handover_rot_left[1],handover_rot_left[2],handover_rot_left[3]]
            pose7 = [target_pos_x,-0.10,0.5,handover_rot_left[0], handover_rot_left[1],handover_rot_left[2],handover_rot_left[3]]
            pose8 = [target_pos_x,0.4,0.5,handover_rot_right[0], handover_rot_right[1],handover_rot_right[2],handover_rot_right[3]]
            pose9 = [target_pos_x,-0.40,0.65,handover_rot_left[0], handover_rot_left[1],handover_rot_left[2],handover_rot_left[3]]
            pose10 = [target_pos_x,-0.30,z+0.05,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose11 = [target_pos_x,-0.30,0.65,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose12 = [target_pos_x,-0.30,0.55,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose13 = [target_pos_x,-0.30,0.65,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose14 = [target_pos_x,-0.30,0.65,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]

            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose5,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose7,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose8,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose9,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose10,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose11,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose12,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose13,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose14,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
            ]
            return pose
    def stack_bottom(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
    


            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
        

            height_range=(-0.05, 0.0)
            dz = np.random.uniform(*height_range)
            
            self.hand_rot,_ = hand_rotate([0.707, 0, 0.707, 0],-self.angle_degree)

            hand_rot= rotate_y([0.707, 0, 0.707, 0],90)
            target_pos_x = 0.5
            target_pos_y = 0
            target_pos_z = z+0.2
            # handover_rot_right = rotate_x([0.707, 0, 0.707, 0],90)
            # handover_rot_right,_ = rotate_z(list(handover_rot_right),90)

            pose1= [x,y,z+0.2,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose2 = [x,y,z+0.05,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose3 = [x,y,z+0.1,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose4 = [target_pos_x,target_pos_y,target_pos_z,hand_rot[0], hand_rot[1],hand_rot[2],hand_rot[3]]
            pose5 = [target_pos_x,target_pos_y,target_pos_z-0.1,hand_rot[0], hand_rot[1],hand_rot[2],hand_rot[3]]
            pose6 = [target_pos_x,target_pos_y,target_pos_z,hand_rot[0], hand_rot[1],hand_rot[2],hand_rot[3]]
           
            if y < 0:
                executor = "robot_right"
                pose7= [x-0.1,y-0.2,z+0.2,hand_rot[0], hand_rot[1],hand_rot[2],hand_rot[3]]
            else:
                executor = "robot_left"
                pose7= [x-0.1,y+0.2,z+0.2,hand_rot[0], hand_rot[1],hand_rot[2],hand_rot[3]]
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose7,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
            ]
            return pose
    
    def stack_middle(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
    


            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
        

            height_range=(-0.05, 0.0)
            dz = np.random.uniform(*height_range)
            
            self.hand_rot,_ = hand_rotate([0.707, 0, 0.707, 0],-self.angle_degree)
            self.hand_rot2,_ = rotate_z([0.707, 0, 0.707, 0],self.angle_degree)

            hand_rot= rotate_y([0.707, 0, 0.707, 0],90)
            target_pos_x = 0.5
            target_pos_y = 0
            target_pos_z = z+0.3
            # handover_rot_right = rotate_x([0.707, 0, 0.707, 0],90)
            # handover_rot_right,_ = rotate_z(list(handover_rot_right),90)

            pose1= [x,y,z+0.2,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose2 = [x,y,z+0.05,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose3 = [x,y,z+0.1,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose4 = [target_pos_x,target_pos_y,target_pos_z,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose5 = [target_pos_x,target_pos_y,target_pos_z-0.2,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose6 = [target_pos_x,target_pos_y,target_pos_z,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]

            if y < 0:
                executor = "robot_right"
                pose7= [x-0.1,y-0.2,z+0.2,hand_rot[0], hand_rot[1],hand_rot[2],hand_rot[3]]
            else:
                executor = "robot_left"
                pose7= [x-0.1,y+0.2,z+0.2,hand_rot[0], hand_rot[1],hand_rot[2],hand_rot[3]]
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose7,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
            ]
            return pose

    def stack_top(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"

            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
    


            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
        

            height_range=(-0.05, 0.0)
            dz = np.random.uniform(*height_range)
            
            self.hand_rot,_ = hand_rotate([0.707, 0, 0.707, 0],-self.angle_degree)
            self.hand_rot2,_ = rotate_z([0.707, 0, 0.707, 0],self.angle_degree)
        

            hand_rot= rotate_y([0.707, 0, 0.707, 0],90)
            target_pos_x = 0.5
            target_pos_y = 0
            target_pos_z = z+0.4
            # handover_rot_right = rotate_x([0.707, 0, 0.707, 0],90)
            # handover_rot_right,_ = rotate_z(list(handover_rot_right),90)

            pose1= [x,y,z+0.2,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose2 = [x,y,z+0.05,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose3 = [x,y,z+0.1,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose4 = [target_pos_x,target_pos_y,target_pos_z,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose5 = [target_pos_x,target_pos_y,target_pos_z-0.2,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
            pose6 = [target_pos_x,target_pos_y,target_pos_z,self.hand_rot[0], self.hand_rot[1],self.hand_rot[2],self.hand_rot[3]]
           
            if y < 0:
                executor = "robot_right"
                pose7= [x-0.1,y-0.2,z+0.2,hand_rot[0], hand_rot[1],hand_rot[2],hand_rot[3]]
            else:
                executor = "robot_left"
                pose7= [x-0.1,y+0.2,z+0.2,hand_rot[0], hand_rot[1],hand_rot[2],hand_rot[3]]
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose7,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
            ]
            return pose

    def on(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
    


            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
        

            height_range=(-0.05, 0.0)
            dz = np.random.uniform(*height_range)
            
            r_x = R.from_euler('x', 90, degrees=True)
            r_y = R.from_euler('y',180, degrees=True)
            # r_z = R.from_euler('z',180, degrees=True)
            r_hand_obj = r_y*r_x    # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            pos_w = [x,y,z]
            pos1 = pos_w - hand_to_world_rot.apply([0.0,0.0,0.0])
            pos2 = pos_w - hand_to_world_rot.apply([0.0,0,0.07])
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)

            pose1= [pos1[0],pos1[1],z+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [pos1[0],pos1[1],z+0.13,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [pos1[0],pos1[1],z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [pos1[0],pos1[1],z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]

        if y > 0:
            executor = "robot_left"
        else:
            executor = "robot_right"
            pose = [
                {"pose":pose1,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},

            ]
            return pose