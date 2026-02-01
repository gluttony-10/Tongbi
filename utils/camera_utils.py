"""
相机控制工具模块

包含相机提示词构建、尺寸更新等功能
"""

from utils.image_utils import snap_to_nearest, calculate_dimensions

# Azimuth mappings (8 positions)
AZIMUTH_MAP = {
    0: "front view",
    45: "front-right quarter view",
    90: "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view"
}

# Elevation mappings (4 positions)
ELEVATION_MAP = {
    -30: "low-angle shot",
    0: "eye-level shot",
    30: "elevated shot",
    60: "high-angle shot"
}

# Distance mappings (3 positions)
DISTANCE_MAP = {
    0.6: "close-up",
    1.0: "medium shot",
    1.8: "wide shot"
}


def build_camera_prompt(azimuth: float, elevation: float, distance: float) -> str:
    """
    Build a camera prompt from azimuth, elevation, and distance values.
    
    Args:
        azimuth: Horizontal rotation in degrees (0-360)
        elevation: Vertical angle in degrees (-30 to 60)
        distance: Distance factor (0.6 to 1.8)
    
    Returns:
        Formatted prompt string for the LoRA
    """
    # Snap to nearest valid values
    azimuth_snapped = snap_to_nearest(azimuth, list(AZIMUTH_MAP.keys()))
    elevation_snapped = snap_to_nearest(elevation, list(ELEVATION_MAP.keys()))
    distance_snapped = snap_to_nearest(distance, list(DISTANCE_MAP.keys()))
    
    azimuth_name = AZIMUTH_MAP[azimuth_snapped]
    elevation_name = ELEVATION_MAP[elevation_snapped]
    distance_name = DISTANCE_MAP[distance_snapped]
    
    return f"<sks> {azimuth_name} {elevation_name} {distance_name}"


def update_dimensions_on_upload_camera(image):
    """Compute recommended dimensions preserving aspect ratio for camera control."""
    if image is None:
        return 1024, 1024, "✅ 根据图片调整宽高"
    
    image_width, image_height = image.size
    vae_width, vae_height = calculate_dimensions(1024*1024, image_width / image_height)
    calculated_height = vae_height // 32 * 32
    calculated_width = vae_width // 32 * 32
    return int(calculated_width), int(calculated_height), "✅ 根据图片调整宽高"
