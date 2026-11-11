from my_utils import *
from set_interface_try import *
import random
ZONES = [
    ((0.45, 0.455), (-0.24, -0.245)),  # zone 0
    ((0.64, 0.645), (-0.08,  -0.085)),  # zone 1
    ((0.40, 0.75), ( 0.30,  0.55)),  # zone 2
]
target_pos_x = np.random.uniform(0.4,0.5)
target_pos_y = np.random.uniform(-0.2,0.2)
target_pos_z = 0.3 

angle_degree = np.random.uniform(0,0.1)

def sample_pos_by_instance(instance_name: str):
    # 从名字里取最后数字，如 my_bowl_bowl_3 -> 3
    idx = int(instance_name.split("_")[-1]) - 1
    xr, yr = ZONES[idx % len(ZONES)]
    x = random.uniform(*xr)
    y = random.uniform(*yr)
    return x, y
class bowl ():
    def __init__(self):
        self.scale = (0.06,0.06,0.06)
        self.target_pos_x = float(target_pos_x)
        self.target_pos_y = float(target_pos_y)
        self.target_pos_z = float(target_pos_z)

    def _stack_place_xyz(self):
        """Shared stack destination. stack_bottom writes the module globals; any bowl can read them."""
        global target_pos_x, target_pos_y, target_pos_z
        tx, ty, tz = float(target_pos_x), float(target_pos_y), float(target_pos_z)
        place = getattr(self, "stack_place", None)
        if place is not None:
            tx, ty = float(place[0]), float(place[1])
        self.target_pos_x = tx
        self.target_pos_y = ty
        self.target_pos_z = tz
        return tx, ty, tz

    def import2sim(
        self,
        urdf_path: str,
        instance_name: str,
        num_envs
    ):
        self.num_envs = num_envs
        
        x1, y1 = sample_pos_by_instance(instance_name)

        
        r_z = R.from_euler('z', angle_degree, degrees=True)
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
            stiffness=0.0,
            damping=0.0,
            fix_base= False,
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
            compliant_contact_stiffness=0
        )

        return [cfg]
    

    def pick(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
        
            print("pick", self.scene_key, "obj_id=", id(self))
            asset = scene[self.actual_instance_name]
            (x, y, z), q_live = live_root_pose_wxyz(asset, env_id)
            print(f"[bowl] live pose {self.actual_instance_name} xyz={[round(x,3), round(y,3), round(z,3)]}")
            r_x = R.from_euler('x', 90, degrees=True)
            r_hand_obj = r_x
            r_obj_world = R.from_quat(wxyz_to_xyzw(q_live))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)

            delta = r_hand_world.apply([0.0, 0.06, 0])
            pos = np.array([x, y, z], dtype=float) + delta

            


            pose1= [pos[0],pos[1],pos[2]+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [pos[0],pos[1],pos[2]+0.07,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [pos[0],pos[1],pos[2]+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [pos[0],pos[1],pos[2]+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
        


            if y < 0:
                executor = "robot_right"
            else:
                executor = "robot_left"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},

            ]
            return pose

    def stack_bottom(self, scene, env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            (x, y, z), q_live = live_root_pose_wxyz(asset, env_id)
            print(f"[bowl] live pose {self.actual_instance_name} xyz={[round(x,3), round(y,3), round(z,3)]}")
            r_x = R.from_euler("x", 90, degrees=True)
            r_hand_world = R.from_quat(wxyz_to_xyzw(q_live)) * r_x
            q_hand_world_wxyz = xyzw_to_wxyz(r_hand_world.as_quat())
            pos = np.array([x, y, z], dtype=float) + r_hand_world.apply([0.0, 0.07, 0])

            global target_pos_x, target_pos_y, target_pos_z
            place = getattr(self, "stack_place", None)
            if place is not None:
                target_pos_x = float(place[0])
                target_pos_y = float(place[1])
            else:
                target_pos_x = float(np.random.uniform(0.55, 0.6))
                target_pos_y = float(np.random.uniform(-0.7, -0.71))
            target_pos_z = float(getattr(self, "stack_table_z", z))
            self.target_pos_x = target_pos_x
            self.target_pos_y = target_pos_y
            self.target_pos_z = target_pos_z
            tx, ty, tz = target_pos_x, target_pos_y, target_pos_z
            print(f"[bowl] stack_bottom place target xyz={[round(tx,3), round(ty,3), round(tz,3)]}")

            q = q_hand_world_wxyz
            pose1 = [pos[0], pos[1], pos[2] + 0.2, q[0], q[1], q[2], q[3]]
            pose2 = [pos[0], pos[1], pos[2] + 0.08, q[0], q[1], q[2], q[3]]
            pose3 = [pos[0], pos[1], pos[2] + 0.08, q[0], q[1], q[2], q[3]]
            pose_lift = [pos[0], pos[1], pos[2] + 0.22, q[0], q[1], q[2], q[3]]
            pose6 = [tx, ty, tz + 0.22, q[0], q[1], q[2], q[3]]
            pose7 = [tx, ty, tz + 0.2, q[0], q[1], q[2], q[3]]
            pose8 = [tx, ty, tz + 0.4, q[0], q[1], q[2], q[3]]
            executor = "robot_right" if y < 0 else "robot_left"
            return [
                {"pose": pose1, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
                {"pose": pose2, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
                {"pose": pose3, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
                {"pose": pose_lift, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
                {"pose": pose6, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
                {"pose": pose7, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
                {"pose": pose8, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
            ]

    def stack_middle(self, scene, env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            (x, y, z), q_live = live_root_pose_wxyz(asset, env_id)
            print(f"[bowl] live pose {self.actual_instance_name} xyz={[round(x,3), round(y,3), round(z,3)]}")
            r_x = R.from_euler("x", 90, degrees=True)
            r_hand_world = R.from_quat(wxyz_to_xyzw(q_live)) * r_x
            q_hand_world_wxyz = xyzw_to_wxyz(r_hand_world.as_quat())
            pos = np.array([x, y, z], dtype=float) + r_hand_world.apply([0.0, 0.07, 0])
            tx, ty, tz = self._stack_place_xyz()
            print(f"[bowl] stack_middle place target xyz={[round(tx,3), round(ty,3), round(tz,3)]}")

            q = q_hand_world_wxyz
            pose1 = [pos[0], pos[1], pos[2] + 0.2, q[0], q[1], q[2], q[3]]
            pose2 = [pos[0], pos[1], pos[2] + 0.08, q[0], q[1], q[2], q[3]]
            pose3 = [pos[0], pos[1], pos[2] + 0.08, q[0], q[1], q[2], q[3]]
            pose_lift = [pos[0], pos[1], pos[2] + 0.25, q[0], q[1], q[2], q[3]]
            pose4 = [tx - 0.03, ty - 0.03, tz + 0.35, q[0], q[1], q[2], q[3]]
            pose5 = [tx - 0.03, ty - 0.03, tz + 0.17, q[0], q[1], q[2], q[3]]
            pose6 = [tx - 0.03, ty - 0.03, tz + 0.4, q[0], q[1], q[2], q[3]]
            executor = "robot_right" if y < 0 else "robot_left"
            return [
                {"pose": pose1, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
                {"pose": pose2, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
                {"pose": pose3, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
                {"pose": pose_lift, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
                {"pose": pose4, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
                {"pose": pose5, "gripper": "close", "use_obstacle_avoidance": False, "executor": executor},
                {"pose": pose6, "gripper": "open", "use_obstacle_avoidance": False, "executor": executor},
            ]

    def stack_bottom_simultaneously_right(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            (x, y, z), q_live = live_root_pose_wxyz(asset, env_id)
            print(f"[bowl] live pose {self.actual_instance_name} xyz={[round(x,3), round(y,3), round(z,3)]}")
            r_x = R.from_euler('x', 90, degrees=True)
            r_hand_obj = r_x
            r_obj_world = R.from_quat(wxyz_to_xyzw(q_live))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)

            delta = r_hand_world.apply([0.0, 0.07, 0])
            pos = np.array([x, y, z], dtype=float) + delta

            self.target_pos_x = float(np.random.uniform(0.55, 0.6))
            self.target_pos_y = float(np.random.uniform(-0.7, -0.71))
            self.target_pos_z = float(z)
            # Shared by stack_middle / stack_top (they used to read the unused module globals).
            global target_pos_x, target_pos_y, target_pos_z
            target_pos_x = self.target_pos_x
            target_pos_y = self.target_pos_y
            target_pos_z = self.target_pos_z
            tx, ty, tz = self.target_pos_x, self.target_pos_y, self.target_pos_z
            print(f"[bowl] stack_bottom place target xyz={[round(tx,3), round(ty,3), round(tz,3)]}")

            pose1= [pos[0],pos[1],pos[2]+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [pos[0],pos[1],pos[2]+0.08,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [pos[0],pos[1],pos[2]+0.08,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose_lift = [pos[0],pos[1],pos[2]+0.22,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            # pose4 = [0.35,-0.33,pos[2]+0.23,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            # pose5 = [0.7,-0.33,pos[2]+0.23,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose6 = [tx,ty,tz+0.22,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose7 = [tx,ty,tz+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose8 = [tx,ty,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]


            if y < 0:
                executor = "robot_right"
                # pose7 = [tx,ty-0.4,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            else:
                executor = "robot_left"
                # pose7 = [tx,ty+0.4,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            executor = "simultaneous_right"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose_lift,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                # {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                # {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose6,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose7,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose8,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                # {"pose":pose7,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},

            ]
            return pose
    
    def stack_middle_simultaneously_right(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            (x, y, z), q_live = live_root_pose_wxyz(asset, env_id)
            print(f"[bowl] live pose {self.actual_instance_name} xyz={[round(x,3), round(y,3), round(z,3)]}")
            r_x = R.from_euler('x', 90, degrees=True)
            r_hand_obj = r_x
            r_obj_world = R.from_quat(wxyz_to_xyzw(q_live))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)

            delta = r_hand_world.apply([0.0, 0.07, 0])
            pos = np.array([x, y, z], dtype=float) + delta

            self.target_pos_x = float(np.random.uniform(0.58, 0.59))
            self.target_pos_y = float(np.random.uniform(-0.7, -0.71))
            self.target_pos_z = float(z)
            tx, ty, tz = self.target_pos_x, self.target_pos_y, self.target_pos_z
            print(f"[bowl] stack_middle place target xyz={[round(tx,3), round(ty,3), round(tz,3)]}")

            pose1= [pos[0],pos[1],pos[2]+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [pos[0],pos[1],pos[2]+0.08,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [pos[0],pos[1],pos[2]+0.08,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose_lift = [pos[0],pos[1],pos[2]+0.25,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [tx-0.03,ty-0.03,tz+0.35,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose5 = [tx-0.03,ty-0.03,tz+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose6 = [tx-0.03,ty-0.03,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]


            
            if y < 0:
                executor = "robot_right"
                # pose7 = [tx,ty-0.4,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                # pose8 = [tx,ty-0.4,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            else:
                executor = "robot_left"
                # pose7 = [tx,ty+0.4,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                # pose8 = [tx,ty+0.4,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            executor = "simultaneous_right"
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose_lift,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                # {"pose":pose7,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                # {"pose":pose8,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},

            ]
            return pose
    
    def stack_top(self,scene,env_id):
        for envid in range(self.num_envs):
            self.actual_instance_name = f"{self.scene_key}__env{envid}"
            asset = scene[self.actual_instance_name]
            (x, y, z), q_live = live_root_pose_wxyz(asset, env_id)
            print(f"[bowl] live pose {self.actual_instance_name} xyz={[round(x,3), round(y,3), round(z,3)]}")
            r_x = R.from_euler('x', 90, degrees=True)
            r_hand_obj = r_x
            r_obj_world = R.from_quat(wxyz_to_xyzw(q_live))
            r_hand_world = r_obj_world * r_hand_obj
            q_hand_world_xyzw = r_hand_world.as_quat()
            q_hand_world_wxyz = xyzw_to_wxyz(q_hand_world_xyzw)

            delta = r_hand_world.apply([0.0, 0.07, 0])
            pos = np.array([x, y, z], dtype=float) + delta
            tx, ty, tz = self._stack_place_xyz()
            print(f"[bowl] stack_top place target xyz={[round(tx,3), round(ty,3), round(tz,3)]}")

            pose1= [pos[0],pos[1],pos[2]+0.2,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose2 = [pos[0],pos[1],pos[2]+0.08,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose3 = [pos[0],pos[1],pos[2]+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose4 = [tx,ty,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose5 = [tx,ty,tz+0.35,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose6 = [tx,ty,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]


            
            if y < 0:
                executor = "robot_right"
                pose7 = [tx,ty-0.4,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                pose8 = [tx,ty-0.4,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            else:
                executor = "robot_left"
                pose7 = [tx,ty+0.4,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
                pose8 = [tx,ty+0.4,tz+0.4,q_hand_world_wxyz[0], q_hand_world_wxyz[1],q_hand_world_wxyz[2],q_hand_world_wxyz[3]]
            pose = [
                {"pose":pose1,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose2,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose3,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose4,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose5,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose6,"gripper":"open","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose7,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},
                {"pose":pose8,"gripper":"close","use_obstacle_avoidance": False,"executor":executor},

            ]
            return pose