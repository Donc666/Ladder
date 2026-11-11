import open3d as o3d
import numpy as np
import os


folder = "/home/jiacheng/7167/textured_objs" 


output_dir = "/home/jiacheng/7167/sim_pointcloud"

# 如果目标文件夹不存在，自动创建它
os.makedirs(output_dir, exist_ok=True)


def sample_mesh(filename, num_points):
    mesh_path = os.path.join(folder, filename)
    if not os.path.exists(mesh_path):
        print(f"警告: 找不到文件 {mesh_path}")
        return np.zeros((0, 3))
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points)


# --- 开始采样 ---
# 1. 采样主体
body_5 = sample_mesh("original_5.obj", 5000)
body_n4 = sample_mesh("new_4.obj", 40000)
body_n3 = sample_mesh("new_3.obj", 10000)
body_n2 = sample_mesh("new_2.obj", 10000)
body_n1 = sample_mesh("new_1.obj", 10000)
body_n0 = sample_mesh("new_0.obj", 10000)
body_pure_pc = np.concatenate([body_5, body_n4, body_n3, body_n2, body_n1, body_n0], axis=0)

# 2. 采样门
door_frame = sample_mesh("original_12.obj", 3000)
door_glass = sample_mesh("original_7.obj", 3000)
door_pure_pc = np.concatenate([door_frame, door_glass], axis=0)

# 3. 采样把手 (link_0 过滤/目标部分)
handle_9 = sample_mesh("original_9.obj", 1000)
handle_10 = sample_mesh("original_10.obj", 1000)
handle_11 = sample_mesh("original_11.obj", 1000)
handle_pure_pc = np.concatenate([handle_9, handle_10, handle_11], axis=0)


body_save_path = os.path.join(output_dir, "body_pure_pc.npy")
door_save_path = os.path.join(output_dir, "door_pure_pc.npy")
handle_save_path = os.path.join(output_dir, "handle_pure_pc.npy")


np.save(body_save_path, body_pure_pc)
np.save(door_save_path, door_pure_pc)
np.save(handle_save_path, handle_pure_pc)
print(f"保存成功！\n主体点云已存至: {body_save_path}\n门点云已存至: {door_save_path}")