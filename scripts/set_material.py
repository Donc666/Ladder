import os
from isaaclab.sim import converters, schemas
from isaaclab.sim.spawners import materials,physics_materials
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg, SpawnerCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.assets import AssetBaseCfg,ArticulationCfg,RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.utils.stage import get_current_stage
import isaacsim.core.utils.physics as physics_utils
from pxr import UsdPhysics, Sdf
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage

def set_physical_material(static_friction,dynamic_friction,restitution,compliant_contact_stiffness,compliant_contact_damping):
    rigid_material_cfg = RigidBodyMaterialCfg(
        static_friction= static_friction,
        dynamic_friction=dynamic_friction ,
        restitution = restitution,
        compliant_contact_stiffness = compliant_contact_stiffness,
        compliant_contact_damping = compliant_contact_damping,
    )
    return rigid_material_cfg

# def spawn_material(object_name:str,static_friction,rigid_material_cfg):
#     prim_path = f"{{ENV_REGEX_NS}}/{object_name}"
#     physical_mat_path = f"{prim_path}/physics_material"
#     rigid_material_cfg.func(prim_path=physical_mat_path, cfg=rigid_material_cfg)
   
#    # 2. 获取当前 Stage 和物体 Prim
#     stage = get_current_stage()
#     object_prim = stage.GetPrimAtPath(prim_path)
    
#     if object_prim.IsValid():
#         # 为物体添加物理材质 API
#         if not object_prim.HasAPI(UsdPhysics.MaterialAPI):
#             usd_phys_mat_api = UsdPhysics.MaterialAPI.Apply(object_prim)
#         else:
#             usd_phys_mat_api = UsdPhysics.MaterialAPI(object_prim)
            
#         # 将材质路径关联到物体的物理材质接口上
#         # 注意：这是物理属性生效的核心步骤
#         usd_phys_mat_api.GetPhysicsMaterialBindingRel().SetTargets([Sdf.Path(physical_mat_path)])
#     else:
#         print(f"Error: 找不到路径 {prim_path}，无法绑定物理材质。")


def spawn_and_bind_material_official(object_name: str, rigid_material_cfg):
    # 1. 遵循官方定义的层级结构
    env_ns = "/World/envs/env_0"
    prim_path = f"{env_ns}/{object_name}"
    # prim_path = f"{{ENV_REGEX_NS}}/{object_name}"
    geom_prim_path = f"{prim_path}/geometry"
    mesh_prim_path = f"{geom_prim_path}/mesh"
    
    # 官方默认将材质放在 geometry 文件夹下
    material_path = f"{geom_prim_path}/physics_material"

    # 2. 调用配置中的 func 来创建材质 Prim (这一步只是生成，不绑定)
    rigid_material_cfg.func(material_path, rigid_material_cfg)
    
    # 3. 获取当前 stage
    stage = get_current_stage()

    # 4. 使用官方接口进行绑定
    # 检查目标 mesh 是否存在，如果不存在则退而求其次绑定到 prim_path
    if stage.GetPrimAtPath(mesh_prim_path).IsValid():
        target_path = mesh_prim_path
    else:
        target_path = prim_path
        
    # 直接调用官方提供的绑定接口
    bind_physics_material(target_path, material_path, stage=stage)