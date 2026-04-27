from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("input_topic", default_value="/cmd_vel_1"),
            DeclareLaunchArgument("output_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument("rule_input_topic", default_value="/traffic_sign/rule_input"),
            DeclareLaunchArgument("use_rule_input_topic", default_value="false"),
            DeclareLaunchArgument(
                "internal_memory_path",
                default_value=(
                    "/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/"
                    "nvdsinfer_custom_impl_Yolo/internal_memory.txt"
                ),
            ),
            Node(
                package="velpub",
                executable="velpub.py",
                name="velpub",
                output="screen",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("input_topic"),
                        "output_topic": LaunchConfiguration("output_topic"),
                        "rule_input_topic": LaunchConfiguration("rule_input_topic"),
                        "use_rule_input_topic": LaunchConfiguration("use_rule_input_topic"),
                        "internal_memory_path": LaunchConfiguration("internal_memory_path"),
                    }
                ],
            )
        ]
    )
