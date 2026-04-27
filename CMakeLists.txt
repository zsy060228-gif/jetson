cmake_minimum_required(VERSION 3.8)
project(velpub)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/detect_line.py
  scripts/detect_line1.py
  scripts/detect_line2.py
  scripts/detect_line3.py
  scripts/detect.sh
  scripts/mock_traffic_sign_detector.py
  scripts/traffic_sign_record.py
  scripts/traffic_sign_replay.py
  scripts/validate_yolov8_config.py
  scripts/velpub.py
  scripts/yolov8_detector.py
  scripts/yolo_rule_adapter.py
  scripts/vth2ros.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  scripts/PID.py
  scripts/Unet.py
  scripts/data.py
  scripts/utils.py
  scripts/lane_local_core.py
  scripts/note.txt
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  scripts/params
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
