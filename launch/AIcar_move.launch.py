from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackagePrefix


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("Speed_stright", default_value="0.16"),
            DeclareLaunchArgument("K_curve", default_value="0.05"),
            DeclareLaunchArgument("Max_angular_curve", default_value="2.6"),
            DeclareLaunchArgument("P", default_value="0.0200"),
            DeclareLaunchArgument("I", default_value="0.0008"),
            DeclareLaunchArgument("D", default_value="0.0030"),
            DeclareLaunchArgument("DeadArea", default_value="2.5"),
            DeclareLaunchArgument("TurnSlowdownGain", default_value="0.70"),
            DeclareLaunchArgument("TurnAngularBoost", default_value="0.40"),
            DeclareLaunchArgument("MinSpeedRatio", default_value="0.40"),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument(
                "note_file",
                default_value=PathJoinSubstitution(
                    [FindPackagePrefix("velpub"), "lib", "velpub", "note.txt"]
                ),
            ),
            Node(
                package="velpub",
                executable="vth2ros.py",
                name="vth2ros",
                output="screen",
                parameters=[
                    {
                        "Speed_stright": LaunchConfiguration("Speed_stright"),
                        "K_curve": LaunchConfiguration("K_curve"),
                        "Max_angular_curve": LaunchConfiguration("Max_angular_curve"),
                        "P": LaunchConfiguration("P"),
                        "I": LaunchConfiguration("I"),
                        "D": LaunchConfiguration("D"),
                        "DeadArea": LaunchConfiguration("DeadArea"),
                        "TurnSlowdownGain": LaunchConfiguration("TurnSlowdownGain"),
                        "TurnAngularBoost": LaunchConfiguration("TurnAngularBoost"),
                        "MinSpeedRatio": LaunchConfiguration("MinSpeedRatio"),
                        "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                        "note_file": LaunchConfiguration("note_file"),
                    }
                ],
            ),
        ]
    )
