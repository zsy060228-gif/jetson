#!/usr/bin/python
# -*- coding:utf-8 


from geometry_msgs.msg import Twist
from std_msgs.msg import Int16
import os
import rospy
#import numpy as np
import time
# from detect_line import last_left,last_right

go_straight = 0
turn_flag = 0
redgreenmark=0
stopmark=0
slowsign=0
slowmark=0
leftsign=0
leftmark=0
redlightmark=0
turnmark=0
crosswalksign=0
upmark = 0
greenmark = 0

#这些都不用改
time2=0.0    #catch turnleft time
time3=0.0    #during turnleft time
time4=0.0    #catch crosswalk stop time
time5=0.0    #catch slowsign time
time6=0.0    #turn to green light time
time7=0.0    #up sign stop
time8=0.0    #jian su 
time9=0.0    #reset_speed
time10=0.0   #green time
time11=0.0   #启动快走

slowedtime=0.0    #time since catched slow sign 


redarea=0.0
turnleft_linear_x=0.18
turnleft_angular_z=1.0
turnleftdelay=4.9
turnningtime=2.1

slowdelay=10.0

redlightdelay=4.18


#delay修改
start_delay = 2.5   #小车启动加速时间
crosswalkdelay=1.25 #小车识别到人行横道后延时这么长时间停车
updelay = 2.4
updelay1 = 2.        #识别到坡道后这么长时间再停
jiansu_delay1 = 3   #红线停止后再次启动时快速变为中速的过渡时间
jiansu_delay2 = 4   #红线停止后再次启动时中速变为慢速的过渡时间
godelay = 8      #红绿灯后延时这么久后停车



resetjiandu_delay = 5.5
green_delay = 5

speed_shift = 3 #3 # 0是停止，1是慢速，2是中速，3是快速,4是左转


#lv add
taskflag1 = 0 #0   #ren wu zhuang tai,mei jing guo yi ge biao zhi pai jia yi
taskflag2 = 1   #ren wu de xia yi ji de ren wu zhuang tai pai xu
sign_save = 0   #0 dai biao no shi bie, 1 dai biao yes shi bie

# def car_stop():
#     msg.linear.x = 0.0
#     msg.angular.z = 0.0

# def car_run(msg_old):
#     msg.linear.x = msg_old.linear.x
#     msg.angular.z = msg_old.angular.z

# def car_run_fast(msg_old):
#     msg.linear.x = msg_old.linear.x * 1.5
#     msg.angular.z = msg_old.angular.z * 1.5

# def car_run_slow(msg_old):
#     msg.linear.x = msg_old.linear.x * 0.7
#     msg.angular.z = msg_old.angular.z * 0.7

# def car_run_left(msg_old):
#     msg.linear.x = msg_old.linear.x
#     msg.angular.z = -5

def flag_reset():#biao zhi wei reset to next task state
    global sign_save,taskflag1,taskflag2
    sign_save = 0
    taskflag2 = 1
    taskflag1 += 1

def left_turnCB(msg):
    global turn_flag, go_straight
    if msg.data == 1:
        turn_flag = 1
        go_straight = 1 
        print("turn_flag:",turn_flag,"go_straight",go_straight) 

def callback(msg_old):
    global redgreenmark, stopmark, slowsign, slowmark, leftsign, leftmark, turnmark, crosswalksign, redlightmark, upmark, greenmark
    global time2, time3, time4, time5, time6, slowedtime, redlightdelay, time7, updelay,updelay1
    global turnleftdelay, turnningtime, crosswalkdelay, slowdelay, error_angular_z, redarea, turnleft_angular_z, slowspeed, turn_flag, turnleft_linear_x
    global speed_shift
    global time8,time9,time10,time11
    global sign_save,taskflag1,taskflag2
    global start_delay

    timenow = time.time()
    # print('timenow:'+str(timenow))

    #读取yolo标签
    txt = open('/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/yolosign.txt', mode='r')
    sign = txt.readline()
    area = txt.readline()
    txt.close()
    

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(50)
    msg = Twist()

    #yolotxt read write conflict
    if(len(sign)!=0 and len(area)!=0):
        # print(sign)
        tmp = list(sign)
        tmp.pop()
        sign = ''.join(tmp)
        print(sign)
        area = float(area)
        #msg.linear.x = msg_old.linear.x
        #msg.angular.z = msg_old.angular.z
        #pub.publish(msg)
        #print('no err')

    print('taskflag1:'+str(taskflag1))
    print('taskflag2:'+str(taskflag2))
    print('sign_save:'+str(sign_save))
    print('sign:'+str(sign))
    # 启动快走
    if taskflag1 == 0:
        if taskflag2 == 1:# start ji shi
            time11 = time.time()
            speed_shift = 3
            taskflag2 += 1
        elif taskflag2 == 2:#减速
            if (timenow-time11)>start_delay:
                speed_shift = 2
                flag_reset()


    # ren xing heng dao
    elif taskflag1 == 1:
        if sign_save == 1:#shi bie yes
            if taskflag2 == 1:# start ji shi
                time4 = time.time()
                speed_shift = 2
                taskflag2 += 1
            elif taskflag2 == 2:#pan duan shi fou stop car
                if (timenow-time4)>crosswalkdelay:
                    speed_shift = 0
                    taskflag2 += 1
            elif taskflag2 == 3:# pan duan shi fou start car
                if (timenow-time4)>(crosswalkdelay+2):
                    speed_shift = 2
                    flag_reset()
        elif sign_save == 0:#shi bie no
            if sign == '0':
                sign_save = 1

    # po lu
    elif taskflag1 == 2:
        if sign_save == 1:
            if taskflag2 == 1:# start ji shi #识别到停车后开始加速
                time7 = time.time()
                speed_shift = 2
                taskflag2 += 1
            elif taskflag2 == 2:#判断是否停车
                if (timenow - time7) > updelay:
                    speed_shift = 0
                    taskflag2 += 1
            elif taskflag2 == 3:#判断是否再次启动小车
                if (timenow - time7) > (updelay + 2.0):
                    speed_shift = 3
                    taskflag2 += 1
            elif taskflag2 == 4:#加速直走一小段距离后减速
                if (timenow - time7) > (updelay + 2.0 + jiansu_delay1):
                    # speed_shift = 2
                    flag_reset()
        elif sign_save == 0:
            if sign == '1':
                sign_save = 1

    # jian su
    elif taskflag1 == 3:
        if taskflag2 == 1:#开始计时
            time8 = time.time()
            speed_shift = 2 
            taskflag2 += 1
        elif taskflag2 == 2:#减速为慢速
            if (timenow - time8) > jiansu_delay2:
                speed_shift = 1
                flag_reset()            
        
    # stop jian su
    elif taskflag1 == 4: 
        if taskflag2 == 1:# start ji shi
            time9 = time.time()
            speed_shift = 1
            taskflag2 += 1
        elif taskflag2 == 2:#5s后恢复中速
            if (timenow - time9) > resetjiandu_delay:
                speed_shift = 2
                flag_reset()      

    # green led
    elif taskflag1 == 5: 
        if sign_save == 1:
            if taskflag2 == 1:# start ji shi
                time7 = time.time()
                speed_shift = 2
                taskflag2 += 1
            elif taskflag2 == 2:#pan duan shi fou stop car
                if (timenow - time7) > updelay1:
                    speed_shift = 0
                    taskflag2 += 1
            elif taskflag2 == 3:# pan duan shi fou start car
                if (timenow - time7) > (updelay1 + 2.0):
                    flag_reset()
        elif sign_save == 0:
            if sign == '2':
                sign_save = 1
                speed_shift = 2

    # go
    elif taskflag1 == 6:
        if sign_save == 1:
            if taskflag2 == 1:# start ji shi
                time7 = time.time()
                speed_shift = 2
                taskflag2 += 1
            elif taskflag2 == 2:#pan duan shi fou stop car
                if (timenow - time7) > godelay:
                    speed_shift = 0
                    taskflag2 += 1
                    flag_reset()
            # elif taskflag2 == 3:# pan duan shi fou start car
            #     if (timenow - time7) > (gpdelay + 2.0):
            #         speed_shift = 2
            #         flag_reset()
        elif sign_save == 0:
            if sign == '3':
                sign_save = 1


    # 根据速度档位发布速度
    if speed_shift == 0:
        msg.linear.x = 0.0
        msg.angular.z = 0.0
    elif speed_shift == 1:
        msg.linear.x = msg_old.linear.x * 0.8
        msg.angular.z = msg_old.angular.z * 0.8
    elif speed_shift == 2:
        msg.linear.x = msg_old.linear.x * 1.2
        msg.angular.z = msg_old.angular.z * 1.2
    elif speed_shift == 3:
        msg.linear.x = msg_old.linear.x * 1.8
        msg.angular.z = msg_old.angular.z * 1.8
    elif speed_shift == 4:
        msg.linear.x = msg_old.linear.x * 0.1
        msg.angular.z = -7
    
    pub.publish(msg)

def subscriber():

    rospy.init_node('veltalker', anonymous=True)
    sub = rospy.Subscriber('/cmd_vel_1',Twist, callback)
    turn_flag_sub = rospy.Subscriber('/turn_flag',Int16, left_turnCB)
    rospy.spin()
    

if __name__ == '__main__':

    try:
        subscriber()
    except rospy.ROSInterruptException:
        pass



# 0.人行横道
# 1.坡道
# 2.左转
# 3.绿灯


