from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("detections_topic", default_value="/traffic_sign/detections"),
            DeclareLaunchArgument("rule_input_topic", default_value="/traffic_sign/rule_input"),
            DeclareLaunchArgument("velpub_input_topic", default_value="/cmd_vel_1"),
            DeclareLaunchArgument("velpub_output_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument("publish_hz", default_value="5.0"),
            DeclareLaunchArgument("loop_sequence", default_value="true"),
            DeclareLaunchArgument(
                "sequence_path",
                default_value="/home/jetson/yahboom_ws/src/velpub/config/mock_traffic_sign_sequence.json",
            ),
            DeclareLaunchArgument(
                "class_map_path",
                default_value="/home/jetson/yahboom_ws/src/velpub/config/yolov8_rule_map.json",
            ),
            DeclareLaunchArgument("confirm_frames", default_value="2"),
            DeclareLaunchArgument("reset_after_sec", default_value="0.75"),
            Node(
                package="velpub",
                executable="mock_traffic_sign_detector.py",
                name="mock_traffic_sign_detector",
                output="screen",
                parameters=[
                    {
                        "detections_topic": LaunchConfiguration("detections_topic"),
                        "publish_hz": LaunchConfiguration("publish_hz"),
                        "loop_sequence": LaunchConfiguration("loop_sequence"),
                        "sequence_path": LaunchConfiguration("sequence_path"),
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
                        "write_internal_memory": False,
                        "confirm_frames": LaunchConfiguration("confirm_frames"),
                        "reset_after_sec": LaunchConfiguration("reset_after_sec"),
                    }
                ],
            ),
            Node(
                package="velpub",
                executable="velpub.py",
                name="velpub",
                output="screen",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("velpub_input_topic"),
                        "output_topic": LaunchConfiguration("velpub_output_topic"),
                        "rule_input_topic": LaunchConfiguration("rule_input_topic"),
                        "use_rule_input_topic": True,
                    }
                ],
            ),
        ]
    )
