

import torch
from isaaclab.utils.math import combine_frame_transforms, quat_apply


def get_asset_id(camera, asset_name: str):
    """获取指定名称 Asset 的语义 ID (适配列表结构的 info 数据)"""
    
    # 检查数据是否存在
    if "semantic_types" not in camera.data.info:
        return None
        
    semantic_types_list = camera.data.info["semantic_types"]
    
    # 因为报错显示这是个 list，我们需要遍历它
    class_type_data = None
    for entry in semantic_types_list:
        # entry 通常是一个字典，包含 'name' 和 'idToLabels'
        if entry.get("name") == "class":
            class_type_data = entry.get("idToLabels")
            break
            
    if class_type_data is None:
        print("警告: 未在相机数据中找到 'class' 语义类型")
        return None
    
    # 在 idToLabels 中查找对应的 asset_name
    # 注意：idToLabels 的 key 有时是字符串类型的数字，所以用 str(sem_id)
    for sem_id, label in class_type_data.items():
        if label == asset_name:
            return int(sem_id)
            
    return None


def get_specific_asset_pointcloud(camera, asset_name: str, env_idx: int = 0):
    """
    获取第 env_idx 个环境中，名为 asset_name 的物体的点云。
    """
    # # 1. 确保开启了语义分割输出
    # if "semantic_seg" not in camera.data.output:
    #     raise RuntimeError("请在相机配置中添加 'semantic_seg' 输出通道")

    # 2. 获取该物体的语义 ID
    asset_id = get_asset_id(camera, asset_name)
    if asset_id is None:
        return torch.empty((0, 3), device=camera.device)

    # 3. 生成全场景点云 [num_envs, N, 3]
    # 使用你之前定义的 get_camera_pointcloud 函数
    points_world_all = get_camera_pointcloud(camera) 
    
    # 4. 获取当前环境的掩码
    # semantic_seg 形状为 [num_envs, H, W]
    semantic_seg = camera.data.output["semantic_seg"][env_idx].reshape(-1)
    mask = (semantic_seg == asset_id)

    # 5. 提取并返回该环境下的 Asset 点云 [M, 3]
    asset_points = points_world_all[env_idx][mask]
    
    return asset_points

def get_camera_pointcloud(camera):
    """获取全场景点云（世界坐标系）"""
    # 1. 获取深度和内参
    depth = camera.data.output["distance_to_image_plane"] # [num_envs, H, W]
    K = camera.data.intrinsic_matrices # [num_envs, 3, 3]
    
    # 缓存像素网格以提高速度
    if not hasattr(camera, "_pixel_grid"):
        h, w = camera.cfg.height, camera.cfg.width
        device = camera.device
        rows, cols = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        camera._pixel_grid = torch.stack([cols.float(), rows.float()], dim=-1) # [H, W, 2]

    u = camera._pixel_grid[..., 0]
    v = camera._pixel_grid[..., 1]

    # 2. 反投影 (修正 USD 坐标系：y 向上，z 向后)
    z = -depth # Isaac Sim 相机看向 -Z 方向
    x = (u - K[:, 0, 2].view(-1, 1, 1)) * depth / K[:, 0, 0].view(-1, 1, 1)
    y = -(v - K[:, 1, 2].view(-1, 1, 1)) * depth / K[:, 1, 1].view(-1, 1, 1) # 取负号修正上下
    
    points_cam = torch.stack([x, y, z], dim=-1).reshape(depth.shape[0], -1, 3)

    # 3. 变换到世界坐标系
    points_world = transform_points(points_cam, camera.data.pos_w, camera.data.quat_w)
    return points_world

def transform_points(
    points: torch.Tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
) -> torch.Tensor:
    """变换点云。
    Args:
        points: 点云，形状为 (..., N, 3)。
        pos: 平移向量，形状为 (..., 3)。默认为 None。
        quat: 旋转四元数 (w, x, y, z)，形状为 (..., 4)。默认为 None。
    Returns:
        变换后的点云，形状为 (..., N, 3)。
    """
    if quat is not None:
        # 这一步利用了你代码里已有的 quat_apply
        points = quat_apply(quat.unsqueeze(-2).repeat(*(1,) * (len(quat.shape) - 1), points.shape[-2], 1), points)
    if pos is not None:
        points = points + pos.unsqueeze(-2)
    return points