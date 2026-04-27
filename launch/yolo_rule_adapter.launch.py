from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("detections_topic", default_value="/traffic_sign/detections"),
            DeclareLaunchArgument("rule_input_topic", default_value="/traffic_sign/rule_input"),
            DeclareLaunchArgument(
                "class_map_path",
                default_value="/home/jetson/yahboom_ws/src/velpub/config/yolov8_rule_map.json",
            ),
            DeclareLaunchArgument(
                "internal_memory_path",
                default_value=(
                    "/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/"
                    "nvdsinfer_custom_impl_Yolo/internal_memory.txt"
                ),
            ),
            DeclareLaunchArgument("write_internal_memory", default_value="true"),
            DeclareLaunchArgument("min_confidence", default_value="0.5"),
            DeclareLaunchArgument("confirm_frames", default_value="2"),
            DeclareLaunchArgument("reset_after_sec", default_value="0.75"),
            Node(
                package="velpub",
                executable="yolo_rule_adapter.py",
                name="yolo_rule_adapter",
                output="screen",
                parameters=[
                    {
                        "detections_topic": LaunchConfiguration("detections_topic"),
                        "rule_input_topic": LaunchConfiguration("rule_input_topic"),
                        "class_map_path": LaunchConfiguration("class_map_path"),
                        "internal_memory_path": LaunchConfiguration("internal_memory_path"),
                        "write_internal_memory": LaunchConfiguration("write_internal_memory"),
                        "min_confidence": LaunchConfiguration("min_confidence"),
                        "confirm_frames": LaunchConfiguration("confirm_frames"),
                        "reset_after_sec": LaunchConfiguration("reset_after_sec"),
                    }
                ],
            ),
        ]
    )
