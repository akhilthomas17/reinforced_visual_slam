#!/usr/bin/env python

import siasainterface as siasa
import rospy
from lsd_slam_viewer.msg import *

from std_msgs.msg import String

def callback(data):
    rospy.loginfo("I heard %s",data.data)
    
def listener():
    rospy.init_node('siasa_viewer')
    rospy.Subscriber("chatter", String, callback)
    print("Initialized siasa viewer node")
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
