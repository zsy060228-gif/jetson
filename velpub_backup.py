#!/usr/bin/python

from geometry_msgs.msg import Twist
import os
import rospy
import numpy as np
import time


redgreenmark=0
stopmark=0
stopmark_2=0
slowsign=0
slowmark=0
slowmark_2=0
leftsign=0
leftmark=0
turnmark=0

time1=0.0    #catch redlight time
time2=0.0    #catch turnleft time
time3=0.0    #during turnleft time
time4=0.0    #catch crosswalk stop time
time5=0.0    #catch slowsign time
time6=0.0    #during slow time



redlightdelay=0.0
turnleftdelay=0.0
turnningtime=0.0
crosswalkdelay=0.0
slowdelay=0.0
errorangularvel=0.0


def callback(msg_old):
    global redgreenmark, stopmark, stopmark_2, slowsign, slowmark, slowmark_2, leftsign, leftmark, turnmark
    global time1, time2, time3, time4, time5, time6
    global redlightdelay, turnleftdelay, turnningtime, crosswalkdelay, slowdelay, errorangularvel
    timenow = time.time()
    txt = open('/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/yolosign.txt', mode='r')
    sign = txt.readline()
    area = txt.readline()
    tmp = list(sign)
    tmp.pop()
    sign = ''.join(tmp)
    txt.close()

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(50)
    msg = Twist()


    #vtherror=-128  straight
    #if msg_old.angular.z>errorangularvel:
        #msg.linear.x = msg_old.linear.x
        #msg.angular.z = 0.1
        #pub.publish(msg)
        

    #yolotxt read write conflict
    if(len(sign)==0):
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        pub.publish(msg)

    

    
    #redlight
    elif sign=='3':
        if redgreenmark==0:
            redgreenmark = 1
            time1=time.time()
        #before stop delay
        elif (timenow-time1)<redlightdelay and redgreenmark==1:
            msg.linear.x = msg_old.linear.x
            msg.angular.z = msg_old.angular.z
            pub.publish(msg)
        #stop
        else:
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            pub.publish(msg)
    
            


    #greenlight
    elif sign=='0' and redgreenmark==1:
        redgreenmark = 0
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        pub.publish(msg)




    #mark left sign
    elif sign=='1' and leftsign==0:
        leftsign=1
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        pub.publish(msg)


    #turnleft
    elif leftsign==1:
        if leftmark==0:
            leftmark = 1
            time2=time.time()
            msg.linear.x = msg_old.linear.x
            msg.angular.z = msg_old.angular.z
            pub.publish(msg)
        #before turn delay
        elif (timenow-time2)<turnleftdelay and leftmark==1:
            msg.linear.x = msg_old.linear.x
            msg.angular.z = msg_old.angular.z
            pub.publish(msg)
        #do turn
        else:
            if turnmark==0:
                turnmark = 1
                time3=time.time()
                msg.linear.x = msg_old.linear.x
                msg.angular.z = msg_old.angular.z
                pub.publish(msg)
            #turnning
            elif (timenow-time3)<turnningtime and turnmark==1:
                msg.linear.x = 0.0
                msg.angular.z = 0.8
                pub.publish(msg)
            else:
                leftsign=0
                msg.linear.x = msg_old.linear.x
                msg.angular.z = msg_old.angular.z
                pub.publish(msg)





    #crosswalk stop  
    elif sign=='4':
        if stopmark==0:
            stopmark = 1
            time4=time.time()
        #before stop delay
        elif (timenow-time4)<crosswalkdelay and stopmark==1:
            msg.linear.x = msg_old.linear.x
            msg.angular.z = msg_old.angular.z
            pub.publish(msg)
        #stop
        elif crosswalkdelay<(timenow-time4)<(crosswalkdelay+1.0) and stopmark==1:
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            pub.publish(msg)
        #go
        else:
            msg.linear.x = msg_old.linear.x
            msg.angular.z = msg_old.angular.z
            pub.publish(msg)
         

    #mark slow sign
    elif sign=='2' and slowsign==0:
        slowsign = 1
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        pub.publish(msg)
        

    #slow
    elif slowsign==1:
        if slowmark==0:
            slowmark==1
            time5=time.time()
        #before slow delay
        elif (timenow-time5)<slowdelay and slowmark==1:
            msg.linear.x = msg_old.linear.x
            msg.angular.z = msg_old.angular.z
            pub.publish(msg)
        #slowdown
        else:
            


            #if slowmark_2==0:
                #slowmark_2=1
                #time6=time.time()
                #msg.linear.x = msg_old.linear.x
                #msg.angular.z = msg_old.angular.z
                #pub.publish(msg)
            #do slow
            #elif (timenow-time6)<5 and slowmark_2==1:
                #msg.linear.x = msg_old.linear.x*0.5
                #msg.angular.z = msg_old.angular.z
                #pub.publish(msg)
            #else:
                #slowsign=0
                #msg.linear.x = msg_old.linear.x
                #msg.angular.z = msg_old.angular.z
                #pub.publish(msg)


    #removeslowlimit
    elif sign=='2':



    #normal
    else:
        msg.linear.x = msg_old.linear.x
        msg.angular.z = msg_old.angular.z
        pub.publish(msg)
        





def subscriber():

    
    rospy.init_node('veltalker', anonymous=True)
    
    sub = rospy.Subscriber('/cmd_vel_1',Twist, callback)
    
    rospy.spin()
    
    



if __name__ == '__main__':

    
    try:
        subscriber()
    except rospy.ROSInterruptException:
        pass







