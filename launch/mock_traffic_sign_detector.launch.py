from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("detections_topic", default_value="/traffic_sign/detections"),
            DeclareLaunchArgument("publish_hz", default_value="5.0"),
            DeclareLaunchArgument("loop_sequence", default_value="true"),
            DeclareLaunchArgument(
                "sequence_path",
                default_value="/home/jetson/yahboom_ws/src/velpub/config/mock_traffic_sign_sequence.json",
            ),
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
        ]
    )
