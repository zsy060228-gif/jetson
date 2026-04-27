from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "profile_path",
                default_value="/home/jetson/yahboom_ws/src/velpub/config/yolov8_project_profile.json",
            ),
            DeclareLaunchArgument("model_path", default_value="/home/jetson/yahboom_ws/best.pt"),
            DeclareLaunchArgument(
                "class_map_path",
                default_value="/home/jetson/yahboom_ws/src/velpub/config/yolov8_rule_map.json",
            ),
            DeclareLaunchArgument("video_device", default_value="/dev/video2"),
            DeclareLaunchArgument("frame_width", default_value="640"),
            DeclareLaunchArgument("frame_height", default_value="480"),
            DeclareLaunchArgument("loop_hz", default_value="10.0"),
            DeclareLaunchArgument("confidence", default_value="0.35"),
            DeclareLaunchArgument("iou", default_value="0.45"),
            DeclareLaunchArgument("max_det", default_value="20"),
            DeclareLaunchArgument("detections_topic", default_value="/traffic_sign/detections"),
            DeclareLaunchArgument("rule_input_topic", default_value="/traffic_sign/rule_input"),
            DeclareLaunchArgument(
                "fallback_video_devices_csv",
                default_value="/dev/video2,/dev/video0,/dev/video1,/dev/video3",
            ),
            DeclareLaunchArgument("stream_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("stream_port", default_value="8091"),
            DeclareLaunchArgument("stream_quality", default_value="80"),
            DeclareLaunchArgument("enable_stream_server", default_value="true"),
            DeclareLaunchArgument("draw_detections", default_value="true"),
            DeclareLaunchArgument(
                "internal_memory_path",
                default_value=(
                    "/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/"
                    "nvdsinfer_custom_impl_Yolo/internal_memory.txt"
                ),
            ),
            DeclareLaunchArgument("write_internal_memory", default_value="true"),
            DeclareLaunchArgument("confirm_frames", default_value="2"),
            DeclareLaunchArgument("reset_after_sec", default_value="0.75"),
            DeclareLaunchArgument("show_debug", default_value="true"),
            Node(
                package="velpub",
                executable="yolov8_detector.py",
                name="yolov8_detector",
                output="screen",
                parameters=[
                    {
                        "profile_path": LaunchConfiguration("profile_path"),
                        "model_path": LaunchConfiguration("model_path"),
                        "class_map_path": LaunchConfiguration("class_map_path"),
                        "video_device": LaunchConfiguration("video_device"),
                        "frame_width": LaunchConfiguration("frame_width"),
                        "frame_height": LaunchConfiguration("frame_height"),
                        "loop_hz": LaunchConfiguration("loop_hz"),
                        "confidence": LaunchConfiguration("confidence"),
                        "iou": LaunchConfiguration("iou"),
                        "max_det": LaunchConfiguration("max_det"),
                        "detections_topic": LaunchConfiguration("detections_topic"),
                        "fallback_video_devices_csv": LaunchConfiguration("fallback_video_devices_csv"),
                        "stream_host": LaunchConfiguration("stream_host"),
                        "stream_port": LaunchConfiguration("stream_port"),
                        "stream_quality": LaunchConfiguration("stream_quality"),
                        "enable_stream_server": LaunchConfiguration("enable_stream_server"),
                        "draw_detections": LaunchConfiguration("draw_detections"),
                        "show_debug": LaunchConfiguration("show_debug"),
                    }
                ],
            ),
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
                        "confirm_frames": LaunchConfiguration("confirm_frames"),
                        "reset_after_sec": LaunchConfiguration("reset_after_sec"),
                    }
                ],
            ),
        ]
    )
