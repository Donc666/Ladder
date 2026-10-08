from my_utils import *
from set_interface_try import *
import random
ZONES = [
    ((0.415, 0.41), (0.36, 0.365)),  # zone left
    ((0.40, 0.42), (-0.09, -0.1)),  # zone right

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
class can ():
    def __init__(self):
        self.scale = (0.04,0.04,0.04)
        pass

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs
    ):
        self.num_envs =num_envs
        self.angle_degree = np.random.uniform(0,0.1)
        x1, y1 = sample_pos_by_instance(instance_name)

        
        r_z = R.from_euler('z', self.angle_degree, degrees=True)
        r_x = R.from_euler('x', 90, degrees=True)
        r_obj = r_z * r_x   
        # self.rot1 若是 wxyz，要先转
        q_obj_world_xyzw = r_obj.as_quat()
        self.q_obj_world_wxyz = xyzw_to_wxyz(q_obj_world_xyzw)
        # rot,self.angel_degree= rotate_z(angle_deg=float(np.random.uniform(-60.0,60.0)),current_quat=[1,0,0,0])
        # rot1,_ = rotate_z([1,0,0,0],40)
        
   

        self.scene_key = instance_name
        print("import", instance_name, "obj_id=", id(self))
        print("pick", self.scene_key, "obj_id=", id(self))

        cfg = AssetCfg(
            
            name = instance_name,
            urdf_path=urdf_path,
            pos = (x1, y1, 0.35),
            rot = self.q_obj_world_wxyz,
            scale = self.scale,
            stiffness=2.0,
            damping=2.0,
            fix_base= False,
            # disable_gravity=True,
            # source_kind = "cuboid",
            cuboid_size= (0.05,0.05,0.05),
            cuboid_color = (1.0, 0.5, 0.2),
            asset_kind = "rigid",
            # diffuse_color=(1.0,0.0,0.0),
            # roughness=0.5,
            metallic=0.9,
            # opacity=1,
            static_friction=0.8,
            dynamic_friction= 0.8,
            restitution= 0,
            compliant_contact_damping=0,
            compliant_contact_stiffness=0,
            mass =0.1
        )

        return [cfg]

    def _pick_waypoints(self, scene, env_id, executor: str):
        """Arrive open at the can, close in place, then lift while staying closed."""
        self.actual_instance_name = f"{self.scene_key}__env{env_id}"
        print("pick", self.scene_key, "obj_id=", id(self))
        asset = scene[self.actual_instance_name]
        try:
            (x, y, z), _ = live_root_pose_wxyz(asset, env_id)
        except Exception:
            root_pos = asset.data.root_pos_w[env_id]
            x, y, z = root_pos[0].item(), root_pos[1].item(), root_pos[2].item()
        print(f"[can] live pose {self.actual_instance_name} xyz={[round(x,3), round(y,3), round(z,3)]}")

        q_hand_world_wxyz = rotate_y([0.707, 0, 0.707, 0], 90)
        q_hand_world_wxyz, _ = rotate_z(list(q_hand_world_wxyz), self.angle_degree)
        q = q_hand_world_wxyz
        pose_pre = [x, y, z + 0.3, q[0], q[1], q[2], q[3]]
        pose_grasp = [x, y, z + 0.13, q[0], q[1], q[2], q[3]]
        pose_lift = [x, y, z + 0.35, q[0], q[1], q[2], q[3]]
        return [
            {"pose": pose_pre, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
            {"pose": pose_grasp, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
            {"pose": pose_grasp, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
            {"pose": pose_lift, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
            {"pose": pose_lift, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
        ]

    def pick(self,scene,env_id):
    
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            try:
                (_, y, _), _ = live_root_pose_wxyz(asset, env_id)
            except Exception:
                y = asset.data.root_pos_w[env_id][1].item()
            executor = "robot_left" if y > 0 else "robot_right"
            return self._pick_waypoints(scene, env_id, executor)
    
    def pick_simultaneously_left(self,scene,env_id):
    
        for envid in range(self.num_envs):
            return self._pick_waypoints(scene, env_id, "simultaneous_left")

    def pick_simultaneously_right(self,scene,env_id):
    
        for envid in range(self.num_envs):
            return self._pick_waypoints(scene, env_id, "simultaneous_right")

    def adjust(self,scene,env_id):
        for envid in range(self.num_envs):
            x, y, z = self._live_root(scene, envid)

            # dl = np.random.uniform(0.05,0.1)
            # x  -=  dl * np.sin(np.deg2rad(self.angle_degree))
            # y  -=  dl * np.cos(np.deg2rad(self.angle_degree))

            target_pos_x = np.random.uniform(0.4,0.45)
            target_pos_y = np.random.uniform(-0.4,-0.)
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

            grasp_pose_pre= [pos1[0],pos1[1],z +0.36,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            grasp_pose = [pos1[0],pos1[1],z+0.06,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            grasp_pose2 = [pos1[0],pos1[1],z+0.36,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose1 = [target_pos_x,target_pos_y,target_pos_z+0.36,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [target_pos_x,target_pos_y,target_pos_z+0.36,0.707, 0, 0.707, 0]
            pose3 = [target_pos_x,target_pos_y,target_pos_z+0.1,0.707, 0, 0.707, 0]
            pose4 = [target_pos_x,target_pos_y,target_pos_z+0.36,0.707, 0, 0.707, 0]
            pose5 = [target_pos_x,target_pos_y,target_pos_z+0.36,0.707, 0, 0.707, 0]
            grasp_pose_pre3= [pos1[0],pos1[1],z +0.26,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]

            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"

            put_pose = [
                {"pose":grasp_pose_pre,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":grasp_pose,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":grasp_pose2,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose1,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose5,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":grasp_pose_pre3,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
            ]