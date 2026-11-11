import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--usd_path", type=str, default="/home/jiacheng/7167/mobility/mobility.usd", help="微波炉 USD 文件的路径")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg,ArticulationCfg,RigidObjectCfg

scale_micwave = 0.3
x3 = 0.2
y3 = 0.2
z3 = 0.2
@configclass
class MySceneCfg(InteractiveSceneCfg):
    # 1. 创建斜面 (静态或固定)
    # 我们使用一个扁平的立方体并将其旋转 30 度
    # slanted_prism = RigidObjectCfg(
    #         prim_path="{ENV_REGEX_NS}/slanted_prism",
    #         spawn=sim_utils.MeshCylinderCfg(
    #             radius=0.5,           # 棱柱中心到顶点的距离
    #             height=2.0,           # 棱柱的长度
    #             num_segments=3,       # 关键：设置为 3 变成三棱柱
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    #             physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
    #             collision_props=sim_utils.CollisionPropertiesCfg(),
    #             rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 必须添加，防止报错
    #         ),
    #         init_state=RigidObjectCfg.InitialStateCfg(
    #             pos=(0.0, 0.0, 0.5),
    #             # 旋转逻辑：
    #             # 1. 绕 X 轴旋转 90 度使棱柱横躺
    #             # 2. 绕自身的轴旋转来控制哪个面朝上（实现“倒放”）
    #             # 3. 绕 Y 轴旋转实现“斜面”坡度
    #             rot=(0.5, 0.5, 0.5, 0.5), # 这是一个示例四元数，你可能需要根据观察微调
    #         ),
    #     )

    # cube1 = Cuboid(
    #     name="mic_bottom",
    #     pose=[x3, y3, z3-0.4*scale_micwave, 1.0, 0.0, 0.0, 0.0], 
    #     dims=[1.0 * scale_micwave, 1.4 * scale_micwave, 0.05 * scale_micwave], # 长, 宽, 高 (单位: 米)
    #     color=[0.4, 0.4, 0.4, 1.0] # 灰色
    # )
    # cube2 = Cuboid(
    #     name="mic_left",
    #     pose=[x3, y3 + 0.8 * scale_micwave, z3, 1.0, 0.0, 0.0, 0.0], 
    #     dims=[1.0 * scale_micwave, 0.05 * scale_micwave, 0.85 * scale_micwave], # 长, 宽, 高 (单位: 米)
    #     color=[0.4, 0.4, 0.4, 1.0] # 灰色
    # )

    # cube3 = Cuboid(
    #     name="mic_right",
    #     pose=[x3, y3 - 0.8 * scale_micwave, z3, 1.0, 0.0, 0.0, 0.0], 
    #     dims=[1.0 * scale_micwave, 0.05 * scale_micwave, 0.85 * scale_micwave], # 长, 宽, 高 (单位: 米)
    #     color=[0.4, 0.4, 0.4, 1.0] # 灰色
    # )
    # cube4 = Cuboid(
    #     name="mic_back",
    #     pose=[x3 + 0.55 * scale_micwave, y3 , z3, 1.0, 0.0, 0.0, 0.0], 
    #     dims=[0.05 * scale_micwave, 1.4 * scale_micwave, 0.85 * scale_micwave], # 长, 宽, 高 (单位: 米)
    #     color=[0.4, 0.4, 0.4, 1.0] # 灰色
    # )
    # cube5 = Cuboid(
    #     name="mic_top",
    #     pose=[x3 , y3 , z3 + 0.7 * scale_micwave, 1.0, 0.0, 0.0, 0.0], 
    #     dims=[1.0 * scale_micwave, 1.4 * scale_micwave, 0.05 * scale_micwave], # 长, 宽, 高 (单位: 米)
    #     color=[0.4, 0.4, 0.4, 1.0] # 灰色
    # )
    # cube6 = Cuboid(
    #     name="mic_front",
    #     pose=[x3 -0.45 * scale_micwave , y3 + 0.6 *scale_micwave, z3, 1.0, 0.0, 0.0, 0.0], 
    #     dims=[0.05 * scale_micwave, 0.2 * scale_micwave, 0.85 * scale_micwave], # 长, 宽, 高 (单位: 米)
    #     color=[0.4, 0.4, 0.4, 1.0] # 灰色
    # )
    # cube7 = Cuboid(
    #     name="floor",
    #     pose=[0 ,0 , -0.05 ,1.0, 0.0, 0.0, 0.0], 
    #     dims=[100, 100, 100], # 长, 宽, 高 (单位: 米)
    #     color=[0.4, 0.4, 0.4, 1.0] # 灰色
    # )
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(0.2,0.6,0.8)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )
    cube1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube1",
        spawn=sim_utils.CuboidCfg(
            size=(1.0 * scale_micwave, 1.4 * scale_micwave, 0.05 * scale_micwave), # 方块大小：5cm x 5cm x 5cm
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.2),roughness=0.2, metallic=1.0 ), # 红色
            rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 赋予刚体属性，使其受重力影响
            collision_props=sim_utils.CollisionPropertiesCfg(), # 赋予碰撞属性，使其能被推开
            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
                pos=(x3, y3, z3-0.4*scale_micwave),
                rot=(1.0, 0.0, 0.0, 0.0),      # 加上四元数
                lin_vel=(0.0, 0.0, 0.0),      # 显式加上线速度
                ang_vel=(0.0, 0.0, 0.0),      # 显式加上角速度
            ),
    )
    cube2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube2",
        spawn=sim_utils.CuboidCfg(
            size=(1.0 * scale_micwave, 0.05 * scale_micwave, 0.85 * scale_micwave), # 方块大小：5cm x 5cm x 5cm
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.2),roughness=0.2, metallic=1.0 ), # 红色
            rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 赋予刚体属性，使其受重力影响
            collision_props=sim_utils.CollisionPropertiesCfg(), # 赋予碰撞属性，使其能被推开
            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
                pos=(x3, y3 + 0.7 * scale_micwave, z3),
                rot=(1.0, 0.0, 0.0, 0.0),      # 加上四元数
                lin_vel=(0.0, 0.0, 0.0),      # 显式加上线速度
                ang_vel=(0.0, 0.0, 0.0),      # 显式加上角速度
            ),
    )
    cube3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube3",
        spawn=sim_utils.CuboidCfg(
            size=(1.0 * scale_micwave, 0.05 * scale_micwave, 0.85 * scale_micwave), # 方块大小：5cm x 5cm x 5cm
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.2),roughness=0.2, metallic=1.0 ), # 红色
            rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 赋予刚体属性，使其受重力影响
            collision_props=sim_utils.CollisionPropertiesCfg(), # 赋予碰撞属性，使其能被推开
            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
                pos=(x3, y3 - 0.7 * scale_micwave, z3),
                rot=(1.0, 0.0, 0.0, 0.0),      # 加上四元数
                lin_vel=(0.0, 0.0, 0.0),      # 显式加上线速度
                ang_vel=(0.0, 0.0, 0.0),      # 显式加上角速度
            ),
    )
    cube4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube4",
        spawn=sim_utils.CuboidCfg(
            size=(0.05 * scale_micwave, 1.4 * scale_micwave, 0.85 * scale_micwave), # 方块大小：5cm x 5cm x 5cm
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.2),roughness=0.2, metallic=1.0 ), # 红色
            rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 赋予刚体属性，使其受重力影响
            collision_props=sim_utils.CollisionPropertiesCfg(), # 赋予碰撞属性，使其能被推开
            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
                pos=(x3 + 0.55 * scale_micwave, y3 , z3),
                rot=(1.0, 0.0, 0.0, 0.0),      # 加上四元数
                lin_vel=(0.0, 0.0, 0.0),      # 显式加上线速度
                ang_vel=(0.0, 0.0, 0.0),      # 显式加上角速度
            ),
    )
    cube5 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube5",
        spawn=sim_utils.CuboidCfg(
            size=(1.0 * scale_micwave, 1.4 * scale_micwave, 0.05 * scale_micwave), # 方块大小：5cm x 5cm x 5cm
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.2),roughness=0.2, metallic=1.0 ), # 红色
            rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 赋予刚体属性，使其受重力影响
            collision_props=sim_utils.CollisionPropertiesCfg(), # 赋予碰撞属性，使其能被推开
            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
                pos=(x3 , y3 , z3 + 0.5 * scale_micwave),
                rot=(1.0, 0.0, 0.0, 0.0),      # 加上四元数
                lin_vel=(0.0, 0.0, 0.0),      # 显式加上线速度
                ang_vel=(0.0, 0.0, 0.0),      # 显式加上角速度
            ),
    )
    cube6 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube6",
        spawn=sim_utils.CuboidCfg(
            size=(0.05 * scale_micwave, 0.2 * scale_micwave, 0.85 * scale_micwave), # 方块大小：5cm x 5cm x 5cm
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.2),roughness=0.2, metallic=1.0 ), # 红色
            rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 赋予刚体属性，使其受重力影响
            collision_props=sim_utils.CollisionPropertiesCfg(), # 赋予碰撞属性，使其能被推开
            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
                pos=(x3 -0.45 * scale_micwave , y3 + 0.6 *scale_micwave, z3),
                rot=(1.0, 0.0, 0.0, 0.0),      # 加上四元数
                lin_vel=(0.0, 0.0, 0.0),      # 显式加上线速度
                ang_vel=(0.0, 0.0, 0.0),      # 显式加上角速度
            ),
    )
        # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
        # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    while simulation_app.is_running():
        # 步进仿真
        sim.step()
        # 更新场景状态（可选，用于同步数据）
        scene.update(dt=sim_cfg.dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()