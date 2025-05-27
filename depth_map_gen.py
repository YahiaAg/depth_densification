import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R

def read_imu_file(file_path):
    """Reads trajectory data from an IMU file."""
    imu_data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            values = line.strip().split()
            if len(values) == 8:
                imu_data.append([float(v) for v in values])
    return np.array(imu_data)

# Transformation matrix: Lidar to Camera (given alignment)
T_imu_to_camera = np.array([
    [-0.0307069, -0.999519, -0.00424704, 0.259398],
    [-0.0640017,  0.00620653, -0.99793,  -0.066651],
    [0.997477, -0.0303716, -0.0641616,  -0.0345687],
    [0, 0, 0, 1]
])


'''T_imu_to_camera = np.array([
   [-0.03871, -0.99922, -0.00802,  0.19196],
    [-0.05322,  0.01024, -0.99853, -0.06529],
    [ 0.99783, -0.03813, -0.05350, -0.34737],
    [ 0,        0,        0,        1     ]
    
])'''


# File paths
imu_file_path = "traj.txt"
pcd_file_path = "scans.pcd"

# Load IMU trajectory
imu_data = read_imu_file(imu_file_path)
if imu_data.shape[1] != 8:
    raise ValueError("IMU file must have 8 columns.")

# Load point cloud
point_cloud = o3d.io.read_point_cloud(pcd_file_path)
if point_cloud.is_empty():
    raise ValueError("PCD file is empty.")
points = np.asarray(point_cloud.points)

# Camera intrinsics
K = np.array([
    [1685.65008, 0, 629.89747],
    [0, 1686.0903, 373.90386],
    [0, 0, 1]
])
image_width, image_height = 1280, 720

# Iterate through IMU trajectory data
for idx, imu_entry in enumerate(imu_data):
    timestamp, x, y, z, q_x, q_y, q_z, q_w = imu_entry

    # Create IMU pose (rotation from quaternion)
    rotation_imu_to_world = R.from_quat([q_x, q_y, q_z, q_w]).as_matrix()
    translation_world = np.array([x, y, z])  # IMU position in world coordinates

    # Construct IMU-to-world transformation
    T_imu_to_world = np.eye(4)
    T_imu_to_world[:3, :3] = rotation_imu_to_world
    T_imu_to_world[:3, 3] = translation_world

    # Compute Camera's position in world coordinates
    t_ic = T_imu_to_camera[:3, 3]  # IMU-to-camera translation
    camera_translation_world = translation_world + rotation_imu_to_world @ t_ic
    print(f"Timestamp: {timestamp:.3f} | Camera Position: X={camera_translation_world[0]:.3f}, Y={camera_translation_world[1]:.3f}, Z={camera_translation_world[2]:.3f}")

    # Compute full transformation: World → Camera = (IMU → Camera) @ (World → IMU)
    T_world_to_imu = np.linalg.inv(T_imu_to_world)  # World → IMU is inverse of IMU → World
    T_world_to_camera = T_imu_to_camera @ T_world_to_imu

    # Transform point cloud into camera coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam = (T_world_to_camera @ points_homogeneous.T).T

    # Filter points in front of the camera
    valid_z = points_cam[:, 2] > 0
    points_valid = points_cam[valid_z, :3]
    if points_valid.size == 0:
        print(f"No valid points for timestamp {timestamp}")
        continue

    # Project points to image plane
    x_proj = points_valid[:, 0] / points_valid[:, 2]
    y_proj = points_valid[:, 1] / points_valid[:, 2]
    u = (K[0, 0] * x_proj + K[0, 2]).astype(int)
    v = (K[1, 1] * y_proj + K[1, 2]).astype(int)

    # Filter valid image coordinates
    valid_uv = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    u_valid = u[valid_uv]
    v_valid = v[valid_uv]
    z_valid = points_valid[valid_uv, 2]

    # Initialize depth map
    depth_map = np.full((image_height, image_width), np.inf)

    # Update depth map with minimum z values
    np.minimum.at(depth_map, (v_valid, u_valid), z_valid)

    # Normalize depth map
    finite_depths = depth_map[np.isfinite(depth_map)]
    if finite_depths.size == 0:
        normalized_depth_map = np.zeros((image_height, image_width), dtype=np.uint8)
    else:
        max_depth = np.max(finite_depths)
        depth_map[~np.isfinite(depth_map)] = max_depth  # Background = max depth
        normalized_depth_map = (depth_map / max_depth * 255).astype(np.uint8)

    # Apply colormap and save depth map
    depth_map_colored = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_INFERNO)
    output_filename = f"depth_map_{timestamp:.3f}.png"
    cv2.imwrite(output_filename, depth_map_colored)
    print(f"Saved: {output_filename}")