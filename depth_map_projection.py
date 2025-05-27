import numpy as np
import open3d as o3d
import cv2
import os
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
    [ 0.997477,  -0.0303716, -0.0641616, -0.0345687],
    [0, 0, 0, 1]
])

# File paths
imu_file_path = "traj.txt"
pcd_file_path = "scans.pcd"
rgb_image_dir = "./images/"  # <-- Set your RGB image folder path here

# Load IMU trajectory
imu_data = read_imu_file(imu_file_path)
if imu_data.shape[1] != 8:
    raise ValueError("IMU file must have 8 columns.")

# Load point cloud
point_cloud = o3d.io.read_point_cloud(pcd_file_path)
if point_cloud.is_empty():
    raise ValueError("PCD file is empty.")
points_all = np.asarray(point_cloud.points)
distances = np.linalg.norm(points_all, axis=1)
points = points_all[distances < 80.0]
# Camera intrinsics
K = np.array([
    [1685.65008, 0, 629.89747],
    [0, 1686.0903, 373.90386],
    [0, 0, 1]
])
image_width, image_height = 1280, 720

# Load RGB image timestamps
rgb_images = []
for filename in os.listdir(rgb_image_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        try:
            ts = float(os.path.splitext(filename)[0])
            rgb_images.append((ts, filename))
        except ValueError:
            continue
rgb_images.sort()


def find_closest_image(timestamp, image_list):
    # Convert IMU timestamp from seconds to nanoseconds
    timestamp_ns = int(timestamp * 1e9)
    # Find the closest image by comparing to the timestamp in nanoseconds
    closest_image = min(image_list, key=lambda x: abs(int(x[0]) - timestamp_ns))
    print(f"Timestamp: {timestamp_ns}, Closest image: {closest_image[1]}")
    return closest_image
# Iterate through IMU trajectory data
for idx, imu_entry in enumerate(imu_data):
    timestamp, x, y, z, q_x, q_y, q_z, q_w = imu_entry

    # Create IMU pose (rotation from quaternion)
    rotation_imu_to_world = R.from_quat([q_x, q_y, q_z, q_w]).as_matrix()
    translation_world = np.array([x, y, z])

    # Construct IMU-to-world transformation
    T_imu_to_world = np.eye(4)
    T_imu_to_world[:3, :3] = rotation_imu_to_world
    T_imu_to_world[:3, 3] = translation_world

    # Compute full transformation: World → Camera = (IMU → Camera) @ (World → IMU)
    T_world_to_imu = np.linalg.inv(T_imu_to_world)
    T_world_to_camera = T_imu_to_camera @ T_world_to_imu

    # Transform point cloud into camera coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam = (T_world_to_camera @ points_homogeneous.T).T

    # Keep only points in front of the camera
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
    np.minimum.at(depth_map, (v_valid, u_valid), z_valid)

    # Normalize and colorize depth map
    finite_depths = depth_map[np.isfinite(depth_map)]
    if finite_depths.size == 0:
        normalized_depth_map = np.zeros((image_height, image_width), dtype=np.uint8)
    else:
        max_depth = np.max(finite_depths)
        depth_map[~np.isfinite(depth_map)] = max_depth
        normalized_depth_map = (depth_map / max_depth * 255).astype(np.uint8)
    depth_map_colored = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_INFERNO)
    # cv2.imwrite(f"depth_map_{timestamp:.3f}.png", depth_map_colored)

    # Find closest RGB image and load it
    closest_ts, closest_filename = find_closest_image(timestamp, rgb_images)
    rgb_path = os.path.join(rgb_image_dir, closest_filename)
    rgb_image = cv2.imread(rgb_path)
    if rgb_image is None:
        print(f"Failed to load image: {rgb_path}")
        continue

    # Sort by depth so closer points are drawn last
    sorted_indices = np.argsort(z_valid)
    u_valid_sorted = u_valid[sorted_indices]
    v_valid_sorted = v_valid[sorted_indices]
    z_valid_sorted = z_valid[sorted_indices]

    # Create an overlay image for drawing points
    overlay = rgb_image.copy()

    # Color points by depth (closer = red, farther = blue)
    depth_colors = (255 * (1 - (z_valid_sorted - np.min(z_valid_sorted)) / (np.max(z_valid_sorted) - np.min(z_valid_sorted)))).astype(np.uint8)

    for i in range(len(u_valid_sorted)):
        color = (int(depth_colors[i]), 0, 255 - int(depth_colors[i]))  # From red to blue
        cv2.circle(overlay, (u_valid_sorted[i], v_valid_sorted[i]), radius=2, color=color, thickness=-1)

    # Blend overlay with the original image
    alpha = 0.5
    blended_image = cv2.addWeighted(overlay, alpha, rgb_image, 1 - alpha, 0)

    # Save the blended overlay
    overlay_filename = f"overlay_{timestamp:.3f}_closest_{closest_ts:.0f}.png"
    cv2.imwrite(overlay_filename, blended_image)
    print(f"Saved overlay image: {overlay_filename}")
