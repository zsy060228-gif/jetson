#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from geometry_msgs.msg import Twist
import os
import time
import rclpy
from rclpy.node import Node
from interfaces.msg import TrafficRuleInput



# 全局变量定义
redgreenmark = 0
stopmark = 0
slowsign = 0
slowmark = 0
leftsign = 0
leftmark = 0
rightsign = 0
rightmark = 0
turnmark = 0
crosswalksign = 0
rampsign = 0
rampmark = 0
noentrysign = 0
noentrymark = 0
noentry_turnmark = 0
peoplesign = 0
peoplemark = 0
emergetime = 0
Slowmark = 0
wait_flag_sign = 0

# 时间记录变量
time2 = 0.0  # 左转开始计时时间
time3 = 0.0  # 左转执行计时时间
time4 = 0.0  # 人行道计时时间 
time5 = 0.0  # 减速标志计时时间
time6 = 0.0  # 进入匝道计时时间
time7 = 0.0  # 驶出匝道计时时间
time8 = 0.0  # 右转开始计时时间
time9 = 0.0  # 右转执行计时时间
time10 = 0.0 # 禁止通行开始计时时间
time11 = 0.0 # 禁止通行执行时间
time12 = 0.0 # 看见行人的时间
time13 = 0.0 # 减速持续时间
slowedtime = 0.0  # 减速持续时间

# 参数配置
redarea = 0
turnleft_angular_z = 1.0
turnright_angular_z  = -1.0
ramp_turnright_angular_z_2 = 0.4 # 出匝道后的第一个右转
turnleftdelay = 0.6 # 左转前等待时间 
turnrightdelay = 0.5 # 右转前等待时间
turnningtime = 1.8  # 转向执行时间
wait_left = 11.8 #第一次左转后等待一段时间左转一点点
wait_right = 12.5 # 出匝道后右转一点点
crosswalkdelay = 0.1 # 识别到人行道后的延迟时间
slowdelay = 1.4  # 减速前等待时间
slowspeed = 0.05  # 减速后的速度
rampduration = 2.0  # 匝道行驶转弯时间
rampdelay = 0.5  # 匝道延迟
in_ramp = 8.0 # 进入匝道行驶的时间
ramp_turnright_angular_z_1 = -1  # 进入匝道右转角速度
noentrydelay = 1.0 # 识别到禁止通行标识后的延迟时间
noentry_turnningtime = 1.2 # 禁止通行的左转时间
no_entry_turnleft_angular_z = 2 # 禁止通行左转角度
peopledelay = 1.0 # 看见行人后的延迟时间 
delay1 = 2.6 #识别到减速后右转前的直行时间
delay2 = 2.2 # 减速的右转时间
slowrightturn = -1.0 # 减速前的右转角速度
slowtimedelay = 24.0 #减速时间


# 状态定义
STATE_NORMAL = 0  # 正常行驶状态
STATE_RED_LIGHT = 1  # 红灯状态
STATE_GREEN_LIGHT = 2  # 绿灯状态
STATE_LEFT_SIGN = 3  # 检测到左转标志状态
STATE_LEFT_TURNING = 4  # 正在左转状态
STATE_CROSSWALK = 5  # 人行道状态
STATE_SLOW_SIGN = 6  # 减速标志状态
STATE_SLOWING = 7  # 正在减速状态
STATE_RAMP_ENTER = 8 # 进入匝道状态
STATE_RAMP_EXIT = 9 # 驶出匝道状态
STATE_RIGHT_SIGN = 10 #检测到右转标志状态
STATE_RIGHT_TURNING = 11  # 正在右转状态
STATE_NO_ENTRY_SIGN = 12 # 检测到禁止通行标识状态
STATE_NO_ENTRY_LEFT_TURNING = 13 # 正在禁止通行的左转状态
STATE_PEOPLE = 14 # 看见行人出现状态

current_state = STATE_NORMAL  # 初始状态为正常行驶
cmd_pub = None
internal_memory_path = '/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/internal_memory.txt'
use_rule_input_topic = False
latest_rule_sign = ""
latest_rule_area = 0.0
latest_rule_stamp = 0.0
rule_input_timeout_sec = 1.0
state_before_people = STATE_NORMAL
last_active_cmd = Twist()
cmd_before_people = Twist()


def clone_twist(src: Twist) -> Twist:
    msg = Twist()
    msg.linear.x = float(src.linear.x)
    msg.linear.y = float(src.linear.y)
    msg.linear.z = float(src.linear.z)
    msg.angular.x = float(src.angular.x)
    msg.angular.y = float(src.angular.y)
    msg.angular.z = float(src.angular.z)
    return msg


def has_motion(src: Twist) -> bool:
    return any(
        abs(value) > 1e-6
        for value in (
            float(src.linear.x),
            float(src.linear.y),
            float(src.linear.z),
            float(src.angular.x),
            float(src.angular.y),
            float(src.angular.z),
        )
    )


def load_rule_input():
    global latest_rule_sign, latest_rule_area, latest_rule_stamp
    global use_rule_input_topic, internal_memory_path, rule_input_timeout_sec
    if use_rule_input_topic:
        if latest_rule_stamp > 0.0 and (time.time() - latest_rule_stamp) > rule_input_timeout_sec:
            latest_rule_sign = ""
            latest_rule_area = 0.0
        return latest_rule_sign, latest_rule_area

    try:
        with open(internal_memory_path, 'r') as txt:
            sign = txt.readline().strip()
            area_line = txt.readline().strip()
        try:
            area = float(area_line) if area_line else 0.0
        except ValueError:
            area = 0.0
        return sign, area
    except Exception:
        return "", 0.0


def callback(msg_old):
    global current_state, redgreenmark, stopmark, slowsign, slowmark, leftsign, leftmark, rightsign, rightmark, turnmark, crosswalksign,noentry_turnmark,noentrymark
    global time2, time3, time4, time5, slowedtime, time6,time11,time10
    global turnleftdelay, turnningtime, crosswalkdelay, slowdelay, redarea, turnleft_angular_z, slowspeed
    global state_before_people, last_active_cmd, cmd_before_people

    timenow = time.time()
    
        
    slowspeed = msg_old.linear.x * 0.7  # 减速速度为原速度的一半

    if current_state != STATE_PEOPLE and has_motion(msg_old):
        last_active_cmd = clone_twist(msg_old)

    # 读取标志检测结果
    sign, area = load_rule_input()

    global cmd_pub
    if cmd_pub is None:
        return
    pub = cmd_pub
    msg = Twist()

    if sign == '9' and current_state != STATE_PEOPLE:
        state_before_people = current_state
        cmd_before_people = clone_twist(msg_old) if has_motion(msg_old) else clone_twist(last_active_cmd)
        current_state = STATE_PEOPLE

    dispatch_state(msg_old, msg, sign, area, timenow, pub)


def dispatch_state(msg_old, msg, sign, area, timenow, pub):
    global current_state

    if current_state == STATE_NORMAL:
        handle_normal_state(msg_old, msg, sign, area, pub)
    elif current_state == STATE_RED_LIGHT:
        handle_red_light_state(msg_old, msg, sign, pub)
    elif current_state == STATE_GREEN_LIGHT:
        handle_green_light_state(msg_old, msg, pub)
    elif current_state == STATE_CROSSWALK:
        handle_crosswalk_state(msg_old, msg, timenow, pub)
    elif current_state == STATE_SLOW_SIGN:
        handle_slow_sign_state(msg_old, msg, sign, timenow, pub)
    elif current_state == STATE_SLOWING:
        handle_slowing_state(msg_old, msg, sign, timenow, pub)
    elif current_state == STATE_RAMP_ENTER:
        handle_ramp_enter_state(msg_old, msg, timenow, pub)
    elif current_state == STATE_RAMP_EXIT:
        handle_ramp_exit_state(msg_old, msg, timenow, pub)
    elif current_state == STATE_NO_ENTRY_SIGN:
        handle_no_entry_sign_state(msg_old, msg, timenow, pub)
    elif current_state == STATE_NO_ENTRY_LEFT_TURNING:
        handle_no_entry_left_turning_state(msg_old, msg, timenow, pub)
    elif current_state == STATE_PEOPLE:
        handle_people_emerge_state(msg_old, msg, sign, area, timenow, pub)
    elif current_state == STATE_RIGHT_SIGN:
        handle_right_sign_state(msg_old, msg, timenow, pub)
    elif current_state == STATE_RIGHT_TURNING:
        handle_right_turning_state(msg_old, msg, timenow, pub)
    elif current_state == STATE_LEFT_SIGN:
        handle_left_sign_state(msg_old, msg, timenow, pub)
    elif current_state == STATE_LEFT_TURNING:
        handle_left_turning_state(msg_old, msg, timenow, pub)
    else:
        current_state = STATE_NORMAL
        handle_normal_state(msg_old, msg, sign, area, pub)


def handle_normal_state(msg_old, msg, sign, area, pub):
    global current_state, redgreenmark, leftsign, crosswalksign, slowsign, rampsign, rightsign, noentrysign, peoplesign, emergetime, time10, time20, wait_left, time3, wait_flag_sign
    temptime = time.time()

    # 正常状态下检查各种交通标志
    if sign == '4' and leftsign == 0 and rightsign == 0:  # 左转标志
        leftsign = 1
        current_state = STATE_LEFT_SIGN
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        print('left sign')
    elif sign == '100' and leftsign == 0 and rightsign == 0:  # 右转标志
        rightsign = 1
        current_state = STATE_RIGHT_SIGN
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        print('right sign')
    elif sign == '5':  # 红灯(and float(area) > redarea:)
        current_state = STATE_RED_LIGHT
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        print('red pub')
    elif sign == '0' and crosswalksign == 0:  # 人行道标志
        crosswalksign = 1
        current_state = STATE_CROSSWALK
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        print('crosswalk_line sign')
    elif sign == '7':  # 减速标志
        slowsign = 1
        current_state = STATE_SLOW_SIGN
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        print('slow sign')
    elif sign == '2' and rampsign != 0:  # 匝道标志
        rampsign = 1
        current_state = STATE_RAMP_ENTER 
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        print('ramp sign')
    elif sign == '8' and noentrysign == 0: # 禁止通行标志
        noentrysign = 1
        current_state = STATE_NO_ENTRY_SIGN
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        print('no entry sign')
    #elif sign == '3': # 人偶标识
        #emergetime += 1
        #if emergetime <= 2:
            #peoplesign = 1
            #current_state = STATE_PEOPLE
            #msg.linear.x = msg_old.linear.x
            #msg.angular.z = msg_old.angular.z
            #print('people emerge')
        #else:
            #msg.linear.x = msg_old.linear.x
            #if msg.angular.z <= 0:
                #msg.angular.z = msg_old.angular.z
            #else:
                #msg.angular.z = msg_old.angular.z * 1.2
            #print('normal pub')
            
    #elif green_sign == 1 and time20 >:
    elif wait_flag_sign==0 and leftsign == 1 and temptime -  time3 > wait_left:
        wait_flag_sign = 1 
        current_state = STATE_LEFT_SIGN
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        print("left after wait")
    else:  # 无特殊标志，正常行驶
        msg.linear.x = msg_old.linear.x
        if msg.angular.z <= 0:
            msg.angular.z = msg_old.angular.z * 1.2
        else:
            msg.angular.z = msg_old.angular.z * 1.2
        print('normal pub')

    pub.publish(msg)


# 处理进入匝道状态
def handle_ramp_enter_state(msg_old, msg, timenow, pub):
    global current_state, time6, rampmark, rampdelay, rampduration, in_ramp, ramp_turnright_angular_z_1

    # 检测到匝道标志后的处理
    if rampmark == 0:
        rampmark = 1
        time6 = timenow
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
    elif (timenow - time6) < rampdelay:  # 进入匝道前的延迟
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        print('before entering ramp')
    elif rampdelay <= (timenow - time6) < (rampdelay + rampduration):  # 在匝道上行驶（右前方）
        msg.linear.x = msg_old.linear.x
        msg.angular.z = ramp_turnright_angular_z_1  # 向右转
        print('on ramp - turning right')
    #elif rampdelay <= (timenow - time6) < (rampdelay + rampduration + in_ramp):
       # msg.linear.x = msg_old.linear.x
       # msg.angular.z = msg_old.angular.z
       # print("in ramp")
    else:  # 准备驶出匝道
        current_state = STATE_NORMAL#STATE_RAMP_EXIT
        rampmark = 0
        print('preparing to exit ramp')

    pub.publish(msg)


# 处理驶出匝道状态
def handle_ramp_exit_state(msg_old, msg, timenow, pub):
    global current_state, time7, rampmark, turnningtime, rampsign, ramp_turnright_angular_z_2

    # 驶出匝道的处理
    if rampmark == 0:
        rampmark = 1
        time7 = timenow
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
    elif (timenow - time7) < turnningtime: # 出匝道后的第一个右转
        msg.linear.x = msg_old.linear.x
        msg.angular.z = ramp_turnright_angular_z_2
        print('exiting ramp - turning right')
    else:  # 匝道行驶完成，返回正常状态
        current_state = STATE_NORMAL
        rampmark = 0
        rampsign = 0
        print('ramp complete, back to normal')

    pub.publish(msg)


def handle_red_light_state(msg_old, msg, sign, pub):
    global current_state

    # 红灯状态下等待绿灯
    if sign == '3':  # 绿灯
        current_state = STATE_GREEN_LIGHT
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        print('green pub')
    else:  # 继续停止
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        print('red pub')

    pub.publish(msg)


def handle_green_light_state(msg_old, msg, pub):
    global current_state

    # 绿灯状态后返回正常状态
    #green_sign = 1
    #time20 = time.time()
    current_state = STATE_NORMAL
    msg.linear.x = msg_old.linear.x
    msg.angular.z = msg_old.angular.z
    print('normal pub')
    pub.publish(msg)


def handle_left_sign_state(msg_old, msg, timenow, pub):
    global current_state, time2, leftmark

    # 检测到左转标志后的处理
    if leftmark == 0:
        leftmark = 1
        time2 = timenow
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
    elif (timenow - time2) < turnleftdelay:  # 左转前的延迟
        msg.linear.x = 0.1
        msg.angular.z = 0.0
        print('before left')
    else:  # 延迟结束，开始左转
        current_state = STATE_LEFT_TURNING
        leftmark = 0  # 重置标记
        print('start left')

    pub.publish(msg)


def handle_left_turning_state(msg_old, msg, timenow, pub):
    global current_state, time3, turnmark, turnleft_angular_z

    # 执行左转动作
    if turnmark == 0:
        turnmark = 1
        time3 = timenow
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
    elif (timenow - time3) < turnningtime:  # 正在左转
        msg.linear.x = 0.0
        msg.angular.z = turnleft_angular_z
        print('turn left')
    #elif (timenow - time3) >= turnningtime and (timenow - time3) < (turnningtime + 8.0):
        #msg.linear.x = msg_old.linear.x
        #msg.angular.z = msg_old.angular.z
        #print('拐弯中')
    #elif (timenow - time3) > (turnningtime + 8.0) and (timenow - time3) < (turnningtime + 10.0):
        #msg.linear.x = 0
        #msg.angular.z = 1.5
        #print('出弯道')
    else:  # 左转完成，返回正常状态
        current_state = STATE_NORMAL
        turnmark = 0

    pub.publish(msg)


def handle_right_sign_state(msg_old, msg, timenow, pub):
    global current_state, time8, rightmark

    # 检测到右转标志后的处理 
    if rightmark == 0:
        rightmark = 1
        time8 = timenow
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
    elif (timenow - time8) < turnrightdelay:  # 右转前的延迟
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        print('before right')
    else:  # 延迟结束，开始右转
        current_state = STATE_RIGHT_TURNING
        rightmark = 0  # 重置标记
        print('start right')

    pub.publish(msg)


def handle_right_turning_state(msg_old, msg, timenow, pub):
    global current_state, time9, turnmark, turnright_angular_z

    # 执行右转动作
    if turnmark == 0:
        turnmark = 1
        time9 = timenow
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
    elif (timenow - time9) < turnningtime:  # 正在右转
        msg.linear.x = 0.0
        msg.angular.z = turnright_angular_z
        print('turn right')
    else:  # 右转完成，返回正常状态
        current_state = STATE_NORMAL
        turnmark = 0

    pub.publish(msg)


def handle_crosswalk_state(msg_old, msg, timenow, pub):
    global current_state, time4, stopmark

    # 人行道状态处理
    if stopmark == 0:
        stopmark = 1
        time4 = timenow
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
    elif (timenow - time4) < crosswalkdelay:  # 人行道前的延迟
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        print('before crosswalk_line stop')
    elif crosswalkdelay < (timenow - time4) and (timenow - time4) < (crosswalkdelay + 4.0):  # 在人行道前停止
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        print('stop')
    elif (timenow - time4) >= (crosswalkdelay + 2.0) and (timenow - time4) < (crosswalkdelay + 3.0):
        msg.linear.x = 0.0
        msg.angular.z = 1.5
    else:  # 停止结束，继续行驶
        current_state = STATE_NORMAL
        stopmark = 0

    pub.publish(msg)


def handle_slow_sign_state(msg_old, msg, sign, timenow, pub):
    global current_state, time5, slowmark, slowspeed, delay1,delay2, slowrightturn, slowdelay

    # 检测到减速标志后的处理
    if slowmark == 0:
        slowmark = 1
        time5 = timenow
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
    elif (timenow - time5) < slowdelay:  # 减速前的延迟
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
    elif (timenow - time5) < (slowdelay + delay1) and (timenow - time5) >= slowdelay:
        msg.linear.x = msg_old.linear.x
        msg.angular.z = 0.0
    elif slowdelay + delay1 <= (timenow - time5) and (timenow - time5) < (slowdelay + delay1 + delay2): # 减速前的转弯
        msg.linear.x = msg_old.linear.x
        msg.angular.z = slowrightturn
        print('slow right turn')
    else:  # 延迟结束，开始减速
        current_state = STATE_SLOWING
        slowmark = 0
        msg.linear.x = 0.1
        msg.angular.z = msg_old.angular.z *0.7
        print('slow pub')


    pub.publish(msg)


def handle_slowing_state(msg_old, msg, sign, timenow, pub):
    global current_state, slowtimedelay, Slowmark, time13

    # 减速行驶状态
    #if sign == '1':  # 解除减速标志
        #current_state = STATE_NORMAL
        #msg.linear.x = msg_old.linear.x
        #msg.angular.z = msg_old.angular.z
        #print('remove slow pub')
    #else:  # 保持减速
        #msg.linear.x = slowspeed
        #msg.angular.z = msg_old.angular.z * 0.7
        #print('slow pub')
    if Slowmark == 0:
        Slowmark = 1
        time13 = timenow
        msg.linear.x = 0.13
        msg.angular.z = msg_old.angular.z * 0.75
        print('slow pub')
    elif (timenow - time13) < slowtimedelay:
        msg.linear.x = 0.13
        msg.angular.z = msg_old.angular.z * 0.75
        print('slow pub')
    else:
        current_state = STATE_NORMAL
        Slowmark = 1
    pub.publish(msg)


def handle_no_entry_sign_state(msg_old, msg, timenow, pub):
    global current_state, time10, noentrymark

    # 检测到禁止通行标志后的处理 
    if noentrymark == 0:
        noentrymark = 1
        time10 = timenow
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
    elif (timenow - time10) < noentrydelay:  # 禁止通行的左转前的延迟
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        print('before no entry left')
    else:  # 延迟结束，开始左转
        current_state = STATE_NO_ENTRY_LEFT_TURNING
        noentrymark = 0  # 重置标记（临时修改）
        print('start no entry left')

    pub.publish(msg)


def handle_no_entry_left_turning_state(msg_old, msg, timenow, pub):
    global current_state, time11, noentry_turnmark, no_entry_turnleft_angular_z

    # 执行禁止通行的左转动作
    if noentry_turnmark == 0:
        noentry_turnmark = 1
        time11 = timenow
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
    elif (timenow - time11) < noentry_turnningtime:  # 正在左转
        msg.linear.x = 0.0
        msg.angular.z = no_entry_turnleft_angular_z
        print('turn no entry left')
    else:  # 左转完成，返回正常状态
        current_state = STATE_NORMAL
        noentry_turnmark = 0
        # 临时修改 
        noentrymark = 0
        

    pub.publish(msg)


def handle_people_emerge_state(msg_old, msg, sign, area, timenow, pub):
    global current_state, time12, peoplemark, state_before_people, cmd_before_people

    # 先检测到行人，进入停车态；只有行人移开后，才恢复之前状态
    if sign == '9':
        if peoplemark == 0:
            peoplemark = 1
            time12 = timenow
            print('people detected, stop')
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        pub.publish(msg)
        return

    restored_state = state_before_people if state_before_people != STATE_PEOPLE else STATE_NORMAL
    resume_cmd = clone_twist(msg_old) if has_motion(msg_old) else clone_twist(cmd_before_people)
    current_state = restored_state
    peoplemark = 0
    time12 = 0.0
    print(f'people cleared, resume state={restored_state}')
    dispatch_state(resume_cmd, msg, sign, area, timenow, pub)

class VelpubNode(Node):
    def __init__(self):
        super().__init__("velpub")
        input_topic = str(self.declare_parameter("input_topic", "/cmd_vel_1").value)
        output_topic = str(self.declare_parameter("output_topic", "/cmd_vel").value)
        rule_input_topic = str(self.declare_parameter("rule_input_topic", "/traffic_sign/rule_input").value)
        declared_use_rule_input_topic = bool(
            self.declare_parameter("use_rule_input_topic", False).value
        )
        declared_internal_memory_path = str(
            self.declare_parameter(
                "internal_memory_path",
                '/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/internal_memory.txt',
            ).value
        )
        declared_rule_input_timeout_sec = float(
            self.declare_parameter("rule_input_timeout_sec", 1.0).value
        )

        global cmd_pub, use_rule_input_topic, internal_memory_path, rule_input_timeout_sec
        cmd_pub = self.create_publisher(Twist, output_topic, 10)
        use_rule_input_topic = declared_use_rule_input_topic
        internal_memory_path = declared_internal_memory_path
        rule_input_timeout_sec = declared_rule_input_timeout_sec
        self.create_subscription(Twist, input_topic, callback, 10)

        if use_rule_input_topic:
            self.create_subscription(TrafficRuleInput, rule_input_topic, self.rule_input_callback, 10)
            self.get_logger().info(
                f"velpub using TrafficRuleInput topic: {rule_input_topic}"
            )
        else:
            self.get_logger().info(
                f"velpub using legacy internal_memory path: {internal_memory_path}"
            )

    def rule_input_callback(self, msg: TrafficRuleInput):
        global latest_rule_sign, latest_rule_area, latest_rule_stamp
        latest_rule_sign = str(msg.legacy_sign_id) if int(msg.legacy_sign_id) >= 0 else ""
        latest_rule_area = float(msg.area_ratio)
        latest_rule_stamp = time.time()


def main(args=None):
    rclpy.init(args=args)
    node = VelpubNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
