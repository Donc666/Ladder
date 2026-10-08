from my_utils import *
from set_interface_try import *
import random

ZONES = [
    ((0.52, 0.525), (-0.35, -0.355)),  # zone 0
    ((0.75, 0.85), (0.1, 0.2)),  # zone 1
    ((0.75, 0.85), (-0.5, -0.10)), 
]
target_pos_x = np.random.uniform(0.4,0.5)
target_pos_y = np.random.uniform(-0.2,0.2)
target_pos_z = 0.3 



def sample_pos_by_instance(instance_name: str):
    # 从名字里取最后数字，如 my_bowl_bowl_3 -> 3
    idx = int(instance_name.split("_")[-1]) - 1
    xr, yr = ZONES[idx % len(ZONES)]
    x = random.uniform(*xr)
    y = random.uniform(*yr)
    return x, y

class bottle ():
    def __init__(self):
        self.scale = (0.08,0.08,0.08)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs
    ):
        
        self.num_envs = num_envs
        self.scene_key = instance_name

        self.angle_degree = np.random.uniform(-130,-131)

        # adjust bottle pose
        # r_z = R.from_euler("z", self.angle_degree, degrees=True)
        # r_x = R.from_euler("x", 0, degrees=True)
        # self.q_obj_world_wxyz = xyzw_to_wxyz((r_z * r_x).as_quat())

        # pick bottle pose
        r_z = R.from_euler('z', self.angle_degree, degrees=True)
        r_x = R.from_euler('x', 90, degrees=True)
        r_obj = r_z * r_x   
        q_obj_world_xyzw = r_obj.as_quat()
        self.q_obj_world_wxyz = xyzw_to_wxyz(q_obj_world_xyzw)
    
        
        # x1 = random.uniform(0.4, 0.8)
        # y1 = random.uniform(-0.5, 0.5)
         # adjust

        #others
        x1, y1 = sample_pos_by_instance(instance_name)
        # x1 = random.uniform(0.4, 0.8)
        # y1 = random.uniform(0.0, 0.5)
        # rot1= rotate_x(current_quat=[1,0,0,0],angle_deg=-90) #not adjust
        
        
        
        
        # rot = rotate_y(list(rot1),-90)




        cfg = AssetCfg(
            fix_base=False,
            name = instance_name,
            urdf_path=urdf_path,
            pos = (x1, y1, 0.32),
            rot = self.q_obj_world_wxyz,
            scale = self.scale,
            stiffness=0.0,
            damping=0.0,
            asset_kind = "rigid",
            # diffuse_color=(1.0,0.0,0.0),
            # roughness=0.5,
            metallic=0.9,
            # opacity=1,
            static_friction=0.8,
            dynamic_friction= 0.8,
            restitution= 0,
            compliant_contact_damping=0,
            compliant_contact_stiffness=0
        )

        return [cfg]

    def _live_root(self, scene, env_id):
        """Read the current scene pose (import-wrapped on first plan, live after a drop)."""
        key = f"{self.scene_key}__env{env_id}"
        try:
            asset = scene[key]
        except Exception:
            asset = scene[self.scene_key]
        pos_buf = asset.data.root_pos_w
        pi = min(int(env_id), int(pos_buf.shape[0]) - 1)
        p = pos_buf[pi]
        quat_buf = getattr(asset.data, "root_quat_w", None)
        if quat_buf is not None:
            q = quat_buf[min(int(env_id), int(quat_buf.shape[0]) - 1)]
            self.q_obj_world_wxyz = [
                float(q[0].item()),
                float(q[1].item()),
                float(q[2].item()),
                float(q[3].item()),
            ]
            self.angle_degree = float(
                R.from_quat(wxyz_to_xyzw(list(self.q_obj_world_wxyz))).as_euler("zyx", degrees=True)[0]
            )
        return float(p[0].item()), float(p[1].item()), float(p[2].item())

    def _import_root(self, scene, env_id):
        """Use the pose from import2sim for atomic-task planning, not the live sim pose."""
        poses = getattr(self, "import_poses", None) or []
        if poses:
            idx = min(int(env_id), len(poses) - 1)
            pos, quat = poses[idx]
            self.q_obj_world_wxyz = list(quat)
            self.angle_degree = float(R.from_quat(wxyz_to_xyzw(list(quat))).as_euler("zyx", degrees=True)[0])
            return float(pos[0]), float(pos[1]), float(pos[2])
        spawn = getattr(self, "spawn_pos", None)
        quat = getattr(self, "import_quat_wxyz", None) or getattr(self, "q_obj_world_wxyz", None)
        if spawn is not None:
            if quat is not None:
                self.q_obj_world_wxyz = list(quat)
            return float(spawn[0]), float(spawn[1]), float(spawn[2])
        key = f"{self.scene_key}__env{env_id}"
        try:
            asset = scene[key]
        except Exception:
            asset = scene[self.scene_key]
        pos_buf = asset.data.root_pos_w
        pi = min(int(env_id), int(pos_buf.shape[0]) - 1)
        p = pos_buf[pi]
        return float(p[0].item()), float(p[1].item()), float(p[2].item())

    def adjust(self,scene,env_id):
        for envid in range(self.num_envs):
            x, y, z = self._live_root(scene, envid)

            # dl = np.random.uniform(0.05,0.1)
            # x  -=  dl * np.sin(np.deg2rad(self.angle_degree))
            # y  -=  dl * np.cos(np.deg2rad(self.angle_degree))

            target_pos_x = np.random.uniform(0.23,0.234)
            target_pos_y = np.random.uniform(-0.69,-0.7)
            target_pos_z = 0.1+ 0.2 
            
            self.hand_rot = rotate_y([0.707, 0, 0.707, 0],90)
            self.hand_rot,_ = rotate_z(current_quat=list(self.hand_rot),angle_deg=self.angle_degree +90)

            r_x = R.from_euler('x', 180, degrees=True)
            r_z = R.from_euler('z',-90, degrees=True)
            r_hand_obj = r_z*r_x    # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            pos_w = [x,y,z]
            pos1 = pos_w - hand_to_world_rot.apply([0.01,0,0.0])
            pos_contact = pos_w - hand_to_world_rot.apply([0.0,0,0.07])
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
        
            
            height_range=(-0.05, 0.0)
            dz = np.random.uniform(*height_range)

            grasp_pose_pre= [pos1[0],pos1[1],z +0.36,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            grasp_pose = [pos1[0],pos1[1],z+0.06,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            grasp_pose2 = [pos1[0],pos1[1],z+0.36,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose1 = [target_pos_x,target_pos_y,target_pos_z+0.36,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [target_pos_x,target_pos_y,target_pos_z+0.09,0.707, 0, 0.707, 0]
            pose3 = [target_pos_x,target_pos_y,target_pos_z+0.09,0.707, 0, 0.707, 0]
            pose4 = [target_pos_x,target_pos_y,target_pos_z+0.09,0.707, 0, 0.707, 0]
            pose5 = [target_pos_x,target_pos_y,target_pos_z+0.09,0.707, 0, 0.707, 0]
            pose6 = [target_pos_x,target_pos_y,target_pos_z+0.3,0.707, 0, 0.707, 0]
            pose7 = [target_pos_x,target_pos_y,target_pos_z+0.3,0.707, 0, 0.707, 0]
  
            executor = "simultaneous_right"

            put_pose = [
                {"pose":grasp_pose_pre,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":grasp_pose,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":grasp_pose2,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose1,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose7,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},

     
            ]
            return put_pose
    def place(self,scene,env_id):
        for envid in range(self.num_envs):
            x, y, z = self._live_root(scene, envid)

            # dl = np.random.uniform(0.05,0.1)
            # x  -=  dl * np.sin(np.deg2rad(self.angle_degree))
            # y  -=  dl * np.cos(np.deg2rad(self.angle_degree))

            target_pos_x = np.random.uniform(0.4,0.45)
            target_pos_y = np.random.uniform(-0.2,-0.3)
            target_pos_z = 0.1+ 0.2 + 0.1 * np.random.uniform(-1,1)
            
            self.hand_rot = rotate_y([0.707, 0, 0.707, 0],90)
            self.hand_rot,_ = rotate_z(current_quat=list(self.hand_rot),angle_deg=self.angle_degree +90)

            r_x = R.from_euler('x', 180, degrees=True)
            r_z = R.from_euler('z',-90, degrees=True)
            r_hand_obj = r_z*r_x    # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            pos_w = [x,y,z]
            pos1 = pos_w - hand_to_world_rot.apply([np.random.uniform(0,0.1),0,0.0])
            pos_contact = pos_w - hand_to_world_rot.apply([0.0,0,0.07])
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)
        
            
            height_range=(-0.05, 0.0)
            dz = np.random.uniform(*height_range)


            pose1= [x-0.2,y,z + dz,0.707, 0, 0.707, 0]
            pose2 = [x-0.09,y,z + dz,0.707, 0, 0.707, 0]
            pose3 = [x-0.09,y,z+0.3,0.707, 0, 0.707, 0]
            pose4 = [x-0.09,y,z+0.3,0.707, 0, 0.707, 0]


            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"

            put_pose = [

                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
    
            ]
            return put_pose
    
    def pick(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]      
    


            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
        

            height_range=(0.06, 0.07)
            dz = np.random.uniform(*height_range)

            target_pos_x = 0.5
            target_pos_y = 0
            target_pos_z = z+0.3
            # self.hand_rot,_ = rotate_z(list(self.hand_rot),self.angel_degree)

            pose1= [x-0.2,y,z + dz,0.707, 0, 0.707, 0]
            pose2 = [x-0.09,y,z + dz,0.707, 0, 0.707, 0]
            pose3 = [x-0.09,y,z+0.3,0.707, 0, 0.707, 0]



            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},


            ]
            return pose
    
    def shake(self,scene,env_id):
        for envid in range(self.num_envs):
            x, y, z = self._live_root(scene, envid)
        

            height_range=(0.08, 0.09)
            dz = np.random.uniform(*height_range)

            target_pos_x = 0.5
            target_pos_y = 0
            target_pos_z = z+0.3
            rot1,_ = rotate_z([0.707, 0, 0.707, 0],20)
            rot2,_ = rotate_z([0.707, 0, 0.707, 0],-20)

            pose1= [x-0.25,y,z + dz,0.707, 0, 0.707, 0]
            pose2 = [x-0.07,y,z + dz,0.707, 0, 0.707, 0]
            pose3 = [x-0.07,y,z+0.3,rot1[0], rot1[1], rot1[2],rot1[3]]
            pose4 = [x-0.07,y,z+0.4,rot2[0], rot2[1], rot2[2],rot2[3]]
            pose5 = [x-0.07,y,z+0.2,rot1[0], rot1[1], rot1[2],rot1[3]]
            pose6 = [x-0.07,y,z+0.2,0.707, 0, 0.707, 0]
            pose7 = [x-0.07,y,z+0.2,0.707, 0, 0.707, 0]
        



            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose6,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose7,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                
    
            ]
            return pose
    
    def shake_horizontally(self,scene,env_id):
        for envid in range(self.num_envs):
            x, y, z = self._live_root(scene, envid)
        

            height_range=(0.07, 0.08)
            dz = np.random.uniform(*height_range)

            target_pos_x = 0.5
            target_pos_y = 0
            target_pos_z = z+0.3
            # self.hand_rot,_ = rotate_z(list(self.hand_rot),self.angel_degree)
            rot1,_= rotate_z(current_quat=[0.707, 0, 0.707, 0],angle_deg=90)
            rot2,_= rotate_z(current_quat=[0.707, 0, 0.707, 0],angle_deg=75)
            rot3,_= rotate_z(current_quat=[0.707, 0, 0.707, 0],angle_deg=105)

            pose1= [x-0.25,y,z + dz,0.707, 0, 0.707, 0]
            pose2 = [x-0.07,y,z + dz,0.707, 0, 0.707, 0]
            pose3 = [x-0.07,y,z+0.3,rot1[0], rot1[1],rot1[2],rot1[3]]
            pose4 = [x-0.07,y,z+0.4,rot2[0],rot2[1],rot2[2], rot2[3]]
            pose5 = [x-0.07,y,z+0.2,rot3[0],rot3[1],rot3[2], rot3[3]]
            pose6 = [x-0.07,y,z+0.2,0.707, 0, 0.707, 0]
            pose7 = [x-0.07,y,z+0.2,0.707, 0, 0.707, 0]
        



            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose6,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose7,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                
    
            ]
            return pose
    def hand_over_left2right(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]   
            
    
            print(f"the asset name :{self.scene_key}the pose got form sim:{root_pos}")

            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
            print(f"the asset name :{self.scene_key}the pose got form sim:{root_pos}")

            height_range=(-0.05, 0.0)
            dz = np.random.uniform(*height_range)
            
            self.hand_rot,_ = hand_rotate([0.707, 0, 0.707, 0],-self.angle_degree)
        

            # handover_rot_left = rotate_x([0.707, 0, 0.707, 0],-90)
            # handover_rot_right = rotate_x([0.707, 0, 0.707, 0],90)
            handover_rot_left = rotate_x([0.707, 0, 0.707, 0],-90)
            handover_rot_right = rotate_x([0.707, 0, 0.707, 0],90)
            x_range = (0,0.3)
            target_pos_x = 0.55
            target_pos_y = 0
            target_pos_z = 0.5

            pose1= [x-0.2,y,z + 0.02,0.707, 0, 0.707, 0]
            pose2 = [x-0.09,y,z+0.02 ,0.707, 0, 0.707, 0]
            pose3 = [x-0.09,y,z+0.4,0.707, 0, 0.707, 0]
            pose4 = [target_pos_x,target_pos_y,target_pos_z,handover_rot_right[0], handover_rot_right[1], handover_rot_right[2], handover_rot_right[3]]
            pose5 = [target_pos_x,target_pos_y-0.3,target_pos_z+0.03,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            pose6 = [target_pos_x,target_pos_y-0.15,target_pos_z+0.03,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            pose7 = [target_pos_x,target_pos_y-0.15,target_pos_z+0.03,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            pose8 = [target_pos_x,target_pos_y,target_pos_z,handover_rot_right[0], handover_rot_right[1], handover_rot_right[2], handover_rot_right[3]]
            pose9 = [target_pos_x,target_pos_y+0.2,target_pos_z+0.03,handover_rot_right[0], handover_rot_right[1], handover_rot_right[2], handover_rot_right[3]]
            pose10 = [target_pos_x,target_pos_y-0.2,target_pos_z+0.03,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            pose11 = [target_pos_x,target_pos_y-0.2,target_pos_z+0.03,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            pose12 = [target_pos_x,target_pos_y-0.2,target_pos_z+0.03,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose5,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose7,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose8,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose9,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose10,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose11,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose12,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},

            ]
            return pose


    def hand_over_right2left(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            root_pos = asset.data.root_pos_w[env_id]   
            
    
            print(f"the asset name :{self.scene_key}the pose got form sim:{root_pos}")

            
            x= root_pos[0].item()
            y = root_pos[1].item()
            z = root_pos[2].item()
            print(f"the asset name :{self.scene_key}the pose got form sim:{root_pos}")

            height_range=(-0.05, 0.0)
            dz = np.random.uniform(*height_range)
            
            self.hand_rot,_ = hand_rotate([0.707, 0, 0.707, 0],-self.angle_degree)
        

            # handover_rot_left = rotate_x([0.707, 0, 0.707, 0],-90)
            # handover_rot_right = rotate_x([0.707, 0, 0.707, 0],90)
            handover_rot_left = rotate_x([0.707, 0, 0.707, 0],-90)
            handover_rot_right = rotate_x([0.707, 0, 0.707, 0],90)
            x_range = (0,0.3)
            target_pos_x = 0.55
            target_pos_y = 0
            target_pos_z = 0.5

            pose1= [x-0.2,y,z + 0.04,0.707, 0, 0.707, 0]
            pose2 = [x-0.09,y,z+0.04 ,0.707, 0, 0.707, 0]
            pose3 = [x-0.09,y,z+0.4,0.707, 0, 0.707, 0]
            pose4 = [target_pos_x,target_pos_y,target_pos_z,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            pose5 = [target_pos_x,target_pos_y+0.2,target_pos_z+0.03,handover_rot_right[0], handover_rot_right[1], handover_rot_right[2], handover_rot_right[3]]
            pose6 = [target_pos_x,target_pos_y+0.1,target_pos_z+0.03,handover_rot_right[0], handover_rot_right[1], handover_rot_right[2], handover_rot_right[3]]
            pose7 = [target_pos_x,target_pos_y+0.1,target_pos_z+0.03,handover_rot_right[0], handover_rot_right[1], handover_rot_right[2], handover_rot_right[3]]
            pose8 = [target_pos_x,target_pos_y,target_pos_z,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            pose9 = [target_pos_x,target_pos_y-0.1,target_pos_z+0.03,handover_rot_left[0], handover_rot_left[1], handover_rot_left[2], handover_rot_left[3]]
            pose10 = [target_pos_x,target_pos_y,target_pos_z+0.03,handover_rot_right[0], handover_rot_right[1], handover_rot_right[2], handover_rot_right[3]]
            pose11 = [target_pos_x,target_pos_y+0.2,target_pos_z+0.03,handover_rot_right[0], handover_rot_right[1], handover_rot_right[2], handover_rot_right[3]]
            pose12 = [target_pos_x,target_pos_y+0.2,target_pos_z+0.03,handover_rot_right[0], handover_rot_right[1], handover_rot_right[2], handover_rot_right[3]]
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose5,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose7,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose8,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose9,"gripper":"open","use_obstacle_avoidance": False,"executor":"robot_right"},
                {"pose":pose10,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose11,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},
                {"pose":pose12,"gripper":"close","use_obstacle_avoidance": False,"executor":"robot_left"},

            ]
            return pose
    


    def pick_simultaneously_left(self,scene,env_id):
        for envid in range(self.num_envs):
            x, y, z = self._live_root(scene, envid)

            self.target_pos_x = np.random.uniform(0.5,0.8)
            self.target_pos_y = np.random.uniform(0.1,0.3)
            self.target_pos_z = z 
        

            height_range=(-0.05, 0.02)
            dz = np.random.uniform(*height_range)

            r_x = R.from_euler('z', 90, degrees=True)
            # r_z = R.from_euler('z', 90, degrees=True)
            r_hand_obj = r_x    # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            pos_w = [x,y,z]
            pos_pre = pos_w - hand_to_world_rot.apply([0.0,0,0.2])
            pos_contact = pos_w - hand_to_world_rot.apply([0.0,0,0.07])
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)

        
            pose1= [pos_pre[0],pos_pre[1],z +0.07,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose2 = [pos_contact[0],pos_contact[1],z +0.07,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose3 = [pos_contact[0],pos_contact[1],z+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose4 = [pos_contact[0],pos_contact[1],z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose5 = [pos_contact[0],pos_contact[1],z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose6 = [pos_contact[0],pos_contact[1],z+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose7 = [self.target_pos_x,self.target_pos_y,self.target_pos_z+0.35,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose8 = [self.target_pos_x,self.target_pos_y,self.target_pos_z+0.1,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose9 = [self.target_pos_x,self.target_pos_y,self.target_pos_z+0.35,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]

                                                                                                                
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose6,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose7,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose8,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
                {"pose":pose9,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_left"},
            ]
            return pose
    
    def pick_simultaneously_right(self,scene,env_id):
        for envid in range(self.num_envs):
            x, y, z = self._live_root(scene, envid)

            self.target_pos_x = np.random.uniform(0.5,0.8)
            self.target_pos_y = np.random.uniform(-0.3,-0.1)
            self.target_pos_z = z 


            r_x = R.from_euler('z', 90, degrees=True)
            # r_z = R.from_euler('z', 90, degrees=True)
            r_hand_obj = r_x    # 先z后x
            # self.rot1 若是 wxyz，要先转
            r_obj_world = R.from_quat(wxyz_to_xyzw(self.q_obj_world_wxyz))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            hand_to_world_rot = R.from_quat(q_hand_world_xyzw)
            pos_w = [x,y,z]
            pos_pre = pos_w - hand_to_world_rot.apply([0.0,0,0.2])
            pos_contact = pos_w - hand_to_world_rot.apply([0.0,0,0.07])
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)

        
            pose1= [pos_pre[0],pos_pre[1],z +0.07,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose2 = [pos_contact[0],pos_contact[1],z +0.07,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose3 = [pos_contact[0],pos_contact[1],z+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose4 = [pos_contact[0],pos_contact[1],z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose5 = [pos_contact[0],pos_contact[1],z+0.3,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose6 = [pos_contact[0],pos_contact[1],z+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose7 = [self.target_pos_x,self.target_pos_y,self.target_pos_z+0.35,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose8 = [self.target_pos_x,self.target_pos_y,self.target_pos_z+0.1,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]
            pose9 = [self.target_pos_x,self.target_pos_y,self.target_pos_z+0.35,q_hand_world_wxyz[0], q_hand_world_wxyz[1], q_hand_world_wxyz[2], q_hand_world_wxyz[3]]




            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose6,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose7,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose8,"gripper":"close","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
                {"pose":pose8,"gripper":"open","use_obstacle_avoidance": False,"executor":"simultaneous_right"},
            
            ]
            return pose
        