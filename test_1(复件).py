#!/home/epaicar/archiconda3/envs/unet/bin/python3


import os

import cv2
import numpy as np
import torch
import time
# from net import *
from Unet import *
from utils import *
from data import *
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt


import os
import socket

import rospy
from geometry_msgs.msg import Twist
import sensor_msgs.msg as smsg
from cv_bridge import CvBridge, CvBridgeError


#'''
#description: 
#return {ip addr}
#'''

count = 0
#time3=time.time()

def get_Ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    return ip


#'''
#description: 
#param {*} img
#param {*} name
#param {*} nwindows
#param {*} margin
#param {*} minpix
#param {*} minLane
#return {*}
#'''

def find_line_fit(img, name = "default" ,nwindows=9, margin=50, minpix=50 , minLane = 50):
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int32(img.shape[1]/2)
    # Set height of windows
    window_height = np.int32(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    h = [0,img.shape[0]]
    w = [0,img.shape[1]]
    leftx_current = w[0]
    rightx_current = w[1]
    # Step through the windows one by one
    for window in range(nwindows):
        start = h[1] - int(h[0] + (h[1] - h[0]) * window / nwindows)
        end = h[1] - int(h[0] + (h[1] - h[0]) * (window + 1) / nwindows)
        histogram = np.sum(img[end:start,w[0]:w[1]], axis=0)

        leftx_current = np.argmax(histogram[:midpoint]) if np.argmax(histogram[:midpoint]) > minLane else leftx_current
        rightx_current = np.argmax(histogram[midpoint:]) + midpoint if np.argmax(histogram[midpoint:]) > minLane else rightx_current

        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        # (0,255,0), 2)
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        # (0,255,0), 2)
        #
        # cv2.line(out_img,(leftx_current,0),(leftx_current,img.shape[1]),(255,0,0))
        # cv2.line(out_img, (rightx_current, 0), (rightx_current, img.shape[1]), (255, 0, 0))
        # cv2.line(out_img, (midpoint, 0), (midpoint, img.shape[1]), (255, 0, 0))
        #
        # cv2.imshow("rec",out_img)
        # cv2.waitKey(0)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        print("error:" + name)
        return [0,0,0],[0,0,0]

    return left_fit, right_fit

def get_angle(img):
    
    cropped = img[180:256,0:256]   #  important  param
    # gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray=cropped
    left, right = find_line_fit(gray)
    bottom_y = int(cropped.shape[0]/2)
    bottom_x_left = int(left[0] * (bottom_y ** 2) + left[1] * bottom_y + left[2])
    bottom_x_right = int(right[0] * (bottom_y ** 2) + right[1] * bottom_y + right[2])

    mid = 128
    #可视化
    cv2.line(cropped, (mid, 0), (mid,cropped.shape[0]), (0, 0, 255), thickness=10)
    cv2.line(cropped,(bottom_x_left,bottom_y),(bottom_x_right,bottom_y),(255,0,0), thickness=10)
    cv2.line(cropped, (mid, bottom_y), (int(bottom_x_left / 2 + bottom_x_right / 2), bottom_y), (0, 255, 0),
            thickness=10)
    
    cv2.imwrite("result/img.jpg",cropped)
    # print(angle)
    angle = int(bottom_x_left / 2 + bottom_x_right / 2) - mid
    return angle

def talker(frame):

    
    rospy.init_node('img_pub', anonymous = True)
    img_pub = rospy.Publisher('/camera/image', smsg.Image, queue_size = 2)
    rate = rospy.Rate(30)
    scaling_factor = 0.5
    bridge = CvBridge()
    
    global count
    if ret:
        count += 1
    else:
        rospy.loginfo("imgpub failed")
    
    if count == 2:
        count = 0
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        msg_1 = bridge.cv2_to_imgmsg(frame,encoding="bgr8")
        img_pub.publish(msg_1)
        print('imgpub')

    #time4=time.time()
    #print(time3)
    #print(time4)
    #print(time4-time3)
    rate.sleep()


if __name__ == "__main__":

    # net=UNet(6).cuda()
    net=UNet().cuda()   #import wyUnet
    # net=nn.DataParallel(net)

    print(next(net.parameters()).device) 
    weights=r'params/unet_0411.pth'
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully')
    else:
        print('no loading')

    _input=r"result/input.png"
    _input_video=r"/home/epaicar/test.mp4"
    #ip=get_Ip()
    #topic="/camera/rgb/image_raw"
    #url="http://"+ip+':8080/stream?topic='+topic




    # img=keep_image_size_open_rgb(_input)
    # # plt.imshow(img)
    # # plt.show()
    # img_data=transform(img).cuda()
    # img_data=torch.unsqueeze(img_data,dim=0)
    # net.eval()
    # time1=time.time()
    # out=net(img_data)
    # # print(out)
    # time2=time.time()
    # print("TIme:",time2-time1)
    # out=torch.argmax(out,dim=1)
    # out=torch.squeeze(out,dim=0)
    # out=out.unsqueeze(dim=0)
    # # print(set((out).reshape(-1).tolist()))
    # out=(out).permute((1,2,0)).cpu().detach().numpy()

    # cv2.imwrite('result/result.png',out*255.0)
    # cv2.imshow('out',out*255.0)
    # cv2.waitKey(0)



    video=cv2.VideoCapture(0)
    #video=cv2.VideoCapture('/home/epaicar/test.mp4')
    ret1=video.set(3,1280)
    ret2=video.set(4,720)
    ret, frame = video.read()
    cv2.imwrite("result/start.jpg",cropped)

    if video.isOpened():
        # video.read() 一帧一帧地读取
        # open 得到的是一个布尔值，就是 True 或者 False
        # frame 得到当前这一帧的图像
        isopen, frame = video.read()
    else:
        isopen = False 
    size=(256, 256)
    while isopen:
    #while isopen:
        ret, frame = video.read()
        #frame = cv2.transpose(frame)

        frame_1 = frame
        rgb_img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # cv2.imshow("rst",frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        
        # frame=cv2.imread(_input)
        heigh=int(frame.shape[0])
        width=int(frame.shape[1])
        # print(width)
        # print(heigh) 
        #裁剪图像，预处理
        frame=frame[int(heigh*2/3):heigh,0:width]
        frame = Image.fromarray(np.uint8(frame))        
        temp = max(frame.size)
        # print(temp)
        mask = Image.new('RGB', (temp, temp))
        # print(frame.size)
        NewWidth=frame.size[0]
        NewHeigh=frame.size[1]
        mask.paste(frame, (0, NewWidth-NewHeigh))
        img = mask.resize(size)
        # img_check = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        # plt.imshow(img)
        # plt.show()
        # img = frame.resize((256,256))


        # img=keep_image_size_open_rgb(_input)
        # plt.imshow(img)
        # plt.show()
        img_data=transform(img).cuda()
        img_data=torch.unsqueeze(img_data,dim=0)
        net.eval()
        time1=time.time()
        ##开始预测
        out=net(img_data)
        # print(out)
        
        
        out=torch.argmax(out,dim=1)
        out=torch.squeeze(out,dim=0)
        out=out.unsqueeze(dim=0)
        # print(set((out).reshape(-1).tolist()))
        out=(out).permute((1,2,0)).cpu().detach().numpy()
        out=out*255.0
        # print(out.shape)
        # vtherror=get_angle(out)
        # print('---------------------------')
        # print(vtherror)
        # out_CHEACK = Image.fromarray(np.uint8(out))
        # s=np.array(out_CHEACK.convert('RGB'))
        # cv2.imwrite('result/result.png',out)
        # Demoframe=cv2.imread('result/result.png')
        # Demoout=out
        # Demoout.shape[2]=3
        # print(Demoout.shape)
        #计算偏差角度
        vtherror=get_angle(out)

        Note=open("note.txt",mode='w')
        Note.write(str(vtherror))
        Note.close()

        talker(frame_1)
        # cv2.imshow("rst",rgb_img)

        # talker.publish(roslibpy.Message({'data':str(vtherror)}))
	
        time2=time.time()
        print('---------------------------')
        print("vtherror:",vtherror)
        print("fps:",1/(time2-time1))
        


