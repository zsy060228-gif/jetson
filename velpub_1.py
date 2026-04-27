from geometry_msgs.msg import Twist
import os
import rospy


def talker():
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.init_node('veltalker', anonymous=True)
    rate = rospy.Rate(50)
    msg = Twist()

    while not rospy.is_shutdown():
        vtherror = txt.read()  # 左负右正？
        vtherror = float(vtherror)
        msg.linear.x = float(0.1)
        msg.angular.z = float((vtherror) * 0.01)
        pub.publish(msg)
        rate.sleep()


if __name__ == '__main__':

    txt = open('note.txt', mode='r')
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

