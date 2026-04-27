#!/usr/bin/python
# -*- coding:utf-8 


from geometry_msgs.msg import Twist
from std_msgs.msg import Int16
import os
import rospy
#import numpy as np
import time
# from detect_line import last_left,last_right

vel_flag = 0
redgreenmark=0
stopmark=0
slowsign=0
slowsign1 = 0
slowmark=0
leftsign=0
leftmark=0
redlightmark=0
turnmark=0
crosswalksign=0
upsign=0
upsign1=0
upmark=0
greenmark = 0
a = 0
b = 0

time2=0.0    #catch turnleft time
time3=0.0    #during turnleft time
time4=0.0    #catch crosswalk stop time
time5=0.0    #catch slowsign time
time6=0.0    #turn to green light time
time7=0.0    #up sign stop
time8=0.0
slowedtime=0.0    #time since catched slow sign 


redarea=0.0
# turnleft_linear_x=0.18
# turnleft_angular_z=1.0
# turnleftdelay=4.9
# turnningtime=2.1
crosswalkdelay=0.1
slowdelay=2.0
updelay = 3.3
redlightdelay=2.75
slowturndelay = 4.5
stop = 7

#error_angular_z=128
slowspeed=0.0


def vel_changeCB(msg):
    global vel_flag
    if msg.data == 1:
        vel_flag = 1
    else:
        vel_flag = 0
        
def callback(msg_old):
    global redgreenmark, stopmark, slowsign, slowmark, leftsign, leftmark, turnmark, crosswalksign, redlightmark,upsign,upsign1,upmark, greenmark, a, b, stop
    global time2, time3, time4, time5, time6, slowedtime, redlightdelay, time7, time8, updelay, slowturndelay
    global turnleftdelay, turnningtime, crosswalkdelay, slowdelay, error_angular_z, redarea, turnleft_angular_z, slowspeed,  turnleft_linear_x

    timenow = time.time()
    slowspeed = msg_old.linear.x*1.2

    txt = open('/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/yolosign.txt', mode='r')
    sign = txt.readline()
    area = txt.readline()
    txt.close()
    

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    #rate = rospy.Rate(50)
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


    #mark crosswalk sign
    if sign=='0' and crosswalksign==0:
        crosswalksign = 1
        msg.linear.x = msg_old.linear.x*2.0
        msg.angular.z = msg_old.angular.z*2.0
        pub.publish(msg)

    #crosswalk stop  
    elif crosswalksign==1 :
        #start timing
        if stopmark==0:
            stopmark = 1
            time4 = time.time()
            msg.linear.x = msg_old.linear.x*1.6
            msg.angular.z = msg_old.angular.z*1.6
        #before stop delay
        elif (timenow-time4)<crosswalkdelay:
            msg.linear.x = msg_old.linear.x*1.6
            msg.angular.z = msg_old.angular.z*1.6
            print('before crosswalk delay')
        #stop
        elif crosswalkdelay<(timenow-time4)<(crosswalkdelay+2.0):
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            print('crosswalk stop')
        # elif timenow-time4<2.0:
        #     msg.linear.x = 0.0
        #     msg.angular.z = 0.0
        #     print('crosswalk stop')
        #go
        else:
            msg.linear.x = msg_old.linear.x *1.6
            msg.angular.z = msg_old.angular.z*1.6
            crosswalksign=2
        pub.publish(msg)

    elif sign=='4' and upsign==0 and crosswalksign==2:
        upsign = 1
        msg.linear.x = msg_old.linear.x*1.6
        msg.angular.z = msg_old.angular.z*1.6
        pub.publish(msg)
        
    #up until remove limit
    elif upsign==1:
        #start timing
        if upmark==0:
            upmark = 1
            time7 = time.time()
            msg.linear.x = msg_old.linear.x*1.6
            msg.angular.z = msg_old.angular.z*1.6
        #before slow delay
        elif (timenow-time7)<updelay:
            msg.linear.x = msg_old.linear.x*1.6
            msg.angular.z = msg_old.angular.z*1.6
            print('before up delay')
       #stop
        elif updelay<(timenow-time7)<(updelay+2.0):
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            print('up stop')
        #go
        else:
            msg.linear.x = msg_old.linear.x *1.7
            msg.angular.z = msg_old.angular.z*1.7
            upsign = 2
        pub.publish(msg)
         
    #mark slow sign
    elif sign=='1' and slowsign==0 and upsign == 2:
        slowsign = 1
        msg.linear.x = msg_old.linear.x*1.6
        msg.angular.z = msg_old.angular.z*1.6
        pub.publish(msg)
        
    #slow until remove limit
    elif slowsign==1:
        #start timing
        if slowmark==0:
            slowmark = 1
            time5 = time.time()
            msg.linear.x = msg_old.linear.x*1.6
            msg.angular.z = msg_old.angular.z*1.6
        #before slow delay
        elif (timenow-time5)<slowdelay:
            msg.linear.x = msg_old.linear.x*1.6
            msg.angular.z = msg_old.angular.z*1.6
            print('before slow delay')
        #remove limit
        elif sign=='2' and (timenow-time5)>slowedtime:
            slowsign = 2
            msg.linear.x = msg_old.linear.x*1.6
            msg.angular.z = msg_old.angular.z*1.6
        #speed down
        else:
            msg.linear.x = slowspeed
            msg.angular.z = msg_old.angular.z*1.2
            print('slow')
        pub.publish(msg)

    elif slowsign == 2:
         #start timing
        if slowmark==1:
            slowmark = 0
            time8 = time.time()
            msg.linear.x = msg_old.linear.x*1.3
            msg.angular.z = msg_old.angular.z*1.3
        elif (timenow-time8)<slowturndelay:
            msg.linear.x = msg_old.linear.x*1.3
            msg.angular.z = msg_old.angular.z*1.3
            print('before slowturn delay')
        elif slowturndelay<(timenow-time8)<(slowturndelay+6.0) :
            msg.linear.x = msg_old.linear.x*1.3
            msg.angular.z = msg_old.angular.z*1.3
            print('slow turn 1')
            a = 1
        elif  12.0<(timenow-time8)<(25.0):
            msg.linear.x = msg_old.linear.x*1.3
            msg.angular.z = msg_old.angular.z*1.3
            # msg.angular.z = -0.85
            print('slow turn 2')
            b = 1
        else:
            msg.linear.x = msg_old.linear.x*1.3
            msg.angular.z = msg_old.angular.z*1.3
            if a == b == 1:
                slowsign = 3
                print(1111111111111111111111111111111111111111111111)
        pub.publish(msg)


    # mark left sign
    elif sign == '3' and leftsign == 0 and slowsign == 3:
        leftsign = 1
        msg.linear.x = msg_old.linear.x*1.3
        msg.angular.z = msg_old.angular.z*1.3
        pub.publish(msg)

    # turnleft
    elif leftsign == 1:
        # start timing
        if leftmark == 0:
            leftmark = 1
            time2 = time.time()
            msg.linear.x = msg_old.linear.x*1.3
            msg.angular.z = msg_old.angular.z*1.3
        # red light
        elif redlightmark == 0:
            # before stop delay
            if (timenow - time2) < redlightdelay:
                msg.linear.x = msg_old.linear.x*1.3
                msg.angular.z = msg_old.angular.z*1.3
            # stop
            else:
                if (timenow - time2) < stop:
                    msg.linear.x = 0.0
                    msg.angular.z = 0.0
                    pub.publish(msg)
                elif sign == '5':
                    msg.linear.x = msg_old.linear.x*1.6
                    msg.angular.z = msg_old.angular.z*1.6
                    # time6 = time.time()
                    redlightmark = 1
                
                
                # green light on
                
        # before turn delay
        else:
            msg.linear.x = msg_old.linear.x*1.6
            msg.angular.z = msg_old.angular.z*1.6
            print('go')

        # else:
        #     if turnmark == 0:
        #         turnmark = 1
        #         time3 = time.time()
        #         msg.linear.x = msg_old.linear.x
        #         msg.angular.z = msg_old.angular.z
        #     # turning
        #     elif (timenow - time3) < turnningtime and turnmark == 1:
        #         msg.linear.x = 0.0
        #         msg.angular.z = turnleft_angular_z
        #         print('turning')
        #     # back to straight
        #     else:
        #         leftsign = 2
        #         msg.linear.x = msg_old.linear.x
        #         msg.angular.z = msg_old.angular.z
        pub.publish(msg)

    #normal
    else:
            msg.linear.x = msg_old.linear.x*1.6
            msg.angular.z = msg_old.angular.z*1.6
            pub.publish(msg)
            print('normal')


def subscriber():

    rospy.init_node('veltalker', anonymous=True)
    sub = rospy.Subscriber('/cmd_vel_1',Twist, callback)
    vel_flag_sub = rospy.Subscriber('/vel_flag',Int16, vel_changeCB)
    rospy.spin()
    

if __name__ == '__main__':

    try:
        subscriber()
    except rospy.ROSInterruptException:
        pass






