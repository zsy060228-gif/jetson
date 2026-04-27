from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackagePrefix


def generate_launch_description():
    laptop_ip = LaunchConfiguration("laptop_ip")
    port = LaunchConfiguration("port")
    video_device = LaunchConfiguration("video_device")
    frame_width = LaunchConfiguration("frame_width")
    frame_height = LaunchConfiguration("frame_height")
    jpeg_quality = LaunchConfiguration("jpeg_quality")
    loop_hz = LaunchConfiguration("loop_hz")
    reconnect_interval_sec = LaunchConfiguration("reconnect_interval_sec")
    linear_speed = LaunchConfiguration("linear_speed")
    steering_gain = LaunchConfiguration("steering_gain")
    publish_cmd_vel = LaunchConfiguration("publish_cmd_vel")
    note_file = LaunchConfiguration("note_file")
    write_note_file = LaunchConfiguration("write_note_file")
    response_floats = LaunchConfiguration("response_floats")

    return LaunchDescription(
        [
            DeclareLaunchArgument("laptop_ip", default_value="10.159.55.120"),
            DeclareLaunchArgument("port", default_value="8000"),
            DeclareLaunchArgument("video_device", default_value="/dev/video0"),
            DeclareLaunchArgument("frame_width", default_value="640"),
            DeclareLaunchArgument("frame_height", default_value="480"),
            DeclareLaunchArgument("jpeg_quality", default_value="80"),
            DeclareLaunchArgument("loop_hz", default_value="20.0"),
            DeclareLaunchArgument("reconnect_interval_sec", default_value="2.0"),
            DeclareLaunchArgument("linear_speed", default_value="0.15"),
            DeclareLaunchArgument("steering_gain", default_value="0.005"),
            DeclareLaunchArgument("publish_cmd_vel", default_value="false"),
            DeclareLaunchArgument("response_floats", default_value="2"),
            DeclareLaunchArgument("yolo_embed_enabled", default_value="true"),
            DeclareLaunchArgument("yolo_stream_port", default_value="8091"),
            DeclareLaunchArgument("yolo_stream_path", default_value="/yolo.mjpg"),
            DeclareLaunchArgument(
                "note_file",
                default_value=PathJoinSubstitution(
                    [FindPackagePrefix("velpub"), "lib", "velpub", "note.txt"]
                ),
            ),
            DeclareLaunchArgument("write_note_file", default_value="true"),
            Node(
                package="velpub",
                executable="detect_line3.py",
                name="detect_line",
                output="screen",
                parameters=[
                    {
                        "laptop_ip": laptop_ip,
                        "port": port,
                        "video_device": video_device,
                        "frame_width": frame_width,
                        "frame_height": frame_height,
                        "jpeg_quality": jpeg_quality,
                        "loop_hz": loop_hz,
                        "reconnect_interval_sec": reconnect_interval_sec,
                        "linear_speed": linear_speed,
                        "steering_gain": steering_gain,
                        "publish_cmd_vel": publish_cmd_vel,
                        "response_floats": response_floats,
                        "note_file": note_file,
                        "write_note_file": write_note_file,
                        "yolo_embed_enabled": LaunchConfiguration("yolo_embed_enabled"),
                        "yolo_stream_port": LaunchConfiguration("yolo_stream_port"),
                        "yolo_stream_path": LaunchConfiguration("yolo_stream_path"),
                    }
                ],
            ),
        ]
    )
