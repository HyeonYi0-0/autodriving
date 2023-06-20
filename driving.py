#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2
import rospy, time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from math import *
import signal
import sys
import os

#=============================================
# 터미널에서 Ctrl-c 키입력이로 프로그램 실행을 끝낼 때
# 그 처리시간을 줄이기 위한 함수
#=============================================
def signal_handler(sig, frame):
    time.sleep(3)
    os.system('killall -9 python rosout')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0]) # 카메라 이미지를 담을 변수
bridge = CvBridge()
motor = None # 모터 토픽을 담을 변수

#=============================================
# 프로그램에서 사용할 상수 선언부
#=============================================
CAM_FPS = 30                # 카메라 FPS - 초당 30장의 사진을 보냄
WIDTH, HEIGHT = 640, 480    # 카메라 이미지 가로x세로 크기
CENTER = 320                # 이미지의 가로 기준 중심

#=============================================
# 콜백함수 - 카메라 토픽을 처리하는 콜백함수
# 카메라 이미지 토픽이 도착하면 자동으로 호출되는 함수
# 토픽에서 이미지 정보를 꺼내 image 변수에 옮겨 담음.
#=============================================
def img_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

#=============================================
# 모터 토픽을 발행하는 함수
# 입력으로 받은 angle과 speed 값을
# 모터 토픽에 옮겨 담은 후에 토픽을 발행함.
#=============================================
def drive(angle, speed):

    global motor

    motor_msg = xycar_motor()
    motor_msg.angle = angle
    motor_msg.speed = speed

    motor.publish(motor_msg)

#==============================================

#=================================================================
# image의 원근감을 조정하는 함수
# 차량에서 받아오는 이미지가 인자이며 이를 마치 위에서 내려다보는 듯한 이미지로 변환함
# 그리고 변환된 이미지를 반환함
#=================================================================
def wrapping(image):
    # version 1: 직선 구간 이미지의 wrapping 결과에서 두 직선 사이의 간격이 넓음 
    # before= [[int(640*0.3), int(480*0.65)], [int(640*0.7), int(480*0.65)],
    #          [int(640*0.05), int(480*0.85)], [int(640*0.95), int(480*0.85)]]

    # after = [[int(640*0.25), int(480*0.2)], [int(640*0.75), int(480*0.2)],
    #          [int(640*0.25), int(480*0.85)], [int(640*0.75), int(480*0.85)]]
    
    # version 2: 직선 구간 이미지의 wrapping 결과에서 두 직선 사이의 간격이 좁음
    # before: (4 X 2) 2차원 배열로 4개 점의 좌표 정보를 담고 있음
    #   각 좌표는 왼쪽 위, 오른쪽 위, 왼쪽 아래, 오른쪽 아래를 의미함
    #   코드에서 직관적으로 알 수 있도록 나타내었음
    before = [[int(WIDTH*0.2), int(HEIGHT*0.65)], [int(WIDTH*0.8), int(HEIGHT*0.65)],
              [int(WIDTH*0), int(HEIGHT*0.85)], [int(WIDTH*1), int(HEIGHT*0.85)]]
    # after: before와 동일하게 4개 점의 좌표 정보를 담고 있음, 각 점은 before가 담고 있는 점에 대응되는 점들임 (왼쪽 위 => 왼쪽 위)
    #   after는 before에서 각 점이 어디로 이동하는 지를 의미함
    #   예: [int(WIDTH*0.2), int(HEIGHT*0.65)](왼쪽 위) ==> [int(WIDTH*0.23), int(HEIGHT*0.2)](왼쪽 위)
    after = [[int(WIDTH*0.23), int(HEIGHT*0.2)], [int(WIDTH*0.77), int(HEIGHT*0.2)],
             [int(WIDTH*0.3), int(HEIGHT*0.85)], [int(WIDTH*0.7), int(HEIGHT*0.85)]]
    
    source = np.float32(before)                                             # before 배열의 요소의 데이터 타입을 float32로 변환하여 source 변수에 할당함
    destination = np.float32(after)                                         # after 배열의 요소의 데이터 타입을 float32로 변환하여 destination 변수에 할당함

    transform_matrix = cv2.getPerspectiveTransform(source, destination)     # source ==> destination, 식으로 나타내면 S(source) = T*D(destination)을 만족하는 T(transform_matrix)을 구함
    _image = cv2.warpPerspective(image, transform_matrix, (WIDTH, HEIGHT))  # transform_matrix를 원본 image에 적용하여 (WIDTH=640, HEIGHT=480) 크기의 변환된 이미지 배열을 반환함
    
    return _image

#=================================================================
# 차선 위치일 가능성이 높은 점을 찾는 함수
# 대략적인 아이디어는 다음과 같다
# 이미지(wrapped_img)에 지평선 기준 아래 부분에 가로선 2개를 그으면 
# 직선 구간의 이미지는 흰색 직선이 세로로 2줄 있으므로 가로선 2개와의 교점 4개가 생긴다
# 그 교점을 각각 upperLeft, upperRight, lowerLeft, lowerRight라고 정의할 수 있다
# 이해를 위해 직선 구간으로 한정해서 설명했으나,
# 직선 구간 뿐만 아니라 곡선 구간, 이미지에서 한 줄 밖에 보이지 않을 때, 가려진 부분이 있을 때는 교점이 생기지 않는데 이때는 교점이 아닌 점은 default 값을 가진다
# 따라서 4개 점이 모두 반환되는 구조이다
#=================================================================
def getLineX(img, horizonY, isStraightLine) :
  leftLimit = CENTER - 5                                        # 이미지 중심을 기준으로 왼쪽 배열을 슬라이싱할 기준을 정함 5는 임의의 값으로 CENTER지점까지 계산할 필요가 없다고 생각해 계산을 줄여주고자 설정한 것임
  rightLimit = CENTER + 5                                       # 이미지 중심을 기준으로 오른쪽 배열을 슬라이싱할 기준을 정함
  
  bottom2horizon = HEIGHT - horizonY                            # 이미지는 위에서 아래로 갈수록 y축 방향 index값이 0~480으로 커짐, 따라서 이미지 밑에서 지평선까지의 높이를 구하기 위한 식임, 즉 bottom2horizon는 밑에서 지평선까지의 높이를 저장함
  upper = int(horizonY + np.round(bottom2horizon*0.1))          # upper는 y축 방향 index값을 저장함, 밑에서 지평선까지의 높이의 10분의 1만큼 더한 값이 지평선 아래에 가로줄을 긋기 위한 지점임
  lower = int(horizonY + np.round(bottom2horizon*0.7))          # lower는 upper와 동일하며 upper보다 더 아래의 가로줄을 긋기 위한 지점 정보를 담음
  upperImgArr = img[upper]                                      # img 배열의 upper index 값 지점에 가로로 이미지의 색상 정보가 1차원 배열로 저장됨 (가로줄1)
  lowerImgArr = img[lower]                                      # 앞서 설명한 것과 동일하게 가로로 이미지의 색상 정보가 1차원 배열로 저장됨 (가로줄2)

  upperLeft = np.argmax(upperImgArr[:leftLimit])                # 위 가로줄 1차원 배열에서 이미지 중심 기준 왼쪽에서 가장 큰 값을 가진 index 값을 찾음, (교점을 찾는 것과 같은 효과)
  upperRight = 640 - np.argmax(upperImgArr[rightLimit:][::-1])  # 오른쪽에서 가장 큰 값을 가진 index 값을 찾음, 이때 slicing하면 index 값이 0 ~320이므로 실제 이미지 배열의 index 값으로 바꾸기 위해서 [::-1] 배열을 역순으로 하여 argmax 인자로 넣은 후 나온 결과값을 640에 뺌
  lowerLeft = np.argmax(lowerImgArr[:leftLimit])                # 아래 가로줄 부분이며 upperLeft 부분 설명과 동일, 교점을 찾지 못하면 index 값은 왼쪽 이므로 0(default)이 됨
  lowerRight = 640 - np.argmax(lowerImgArr[rightLimit:][::-1])  # upperRight 부분 설명과 동일, 교점을 찾지 못하면 index 값은 오른쪽이라 640(default)이 됨
  
  upLen = upperRight - upperLeft                                # 위 가로줄의 왼쪽 오른쪽 값 사이의 거리를 구함
  lowLen = lowerRight - lowerLeft                               # 아래 가로줄의 왼쪽 오른쪽 값 사이의 거리를 구함
  
  if isStraightLine :                                           # 직선 구간이면 
    upperRight = 640 - upperLeft                                # 각 지점들이 320을 기준으로 대칭이 되도록 함, 이는 angle을 직선으로 만들기 위함
    lowerRight = 640 - lowerLeft
  elif (upperLeft == 0 and upperRight != 640 and lowerLeft != 0 and lowerRight != 640) : # 3개 감지 왼쪽 상단 감지 X
    upperLeft = upperRight - lowLen                                                      # 감지되지 않은 값은 나머지 3개의 값으로 추론함 왼쪽 상단 값을 모르므로 오른쪽 상단 값에서 아래 두 지점 사이의 거리 만큼을 빼면 됨
  elif (upperLeft != 0 and upperRight == 640 and lowerLeft != 0 and lowerRight != 640) : # 3개 감지 오른쪽 상단 감지 X
    upperRight = upperLeft + lowLen
  elif (upperLeft != 0 and upperRight != 640 and lowerLeft == 0 and lowerRight != 640) : # 3개 감지 왼쪽 하단 감지 X
    lowerLeft = lowerRight - upLen
  elif (upperLeft != 0 and upperRight != 640 and lowerLeft != 0 and lowerRight == 640) : # 3개 감지 오른쪽 하단 감지 X
    lowerRight = lowerLeft + upLen
  elif (upperLeft != 0 and lowerLeft != 0 and upperRight == 640 and lowerRight == 640) : # 2개 감지 왼쪽 상단과 왼쪽 하단 감지
    mid = (upperLeft + lowerLeft) // 2                                                   # 왼쪽 상단 지점(x축 방향 값)과 왼쪽 하단 지점 값의 평균을 구함
    if mid < (leftLimit-mid) :                                                           # 차선이 왼쪽으로 너무 치우쳐져 있을 때 (0 ~ mid ~ leftLimit)인데 왼쪽으로 치우치면 각 지점 사이의 거리에서 0과 mid 사이의 거리가 mid와 leftLimit 사이의 거리보다 작음
      upperRight = leftLimit                                                             # 차선이 왼쪽이 치우쳐져 감지된 것은 차량이 오른쪽에 붙었음을 의미, 따라서 왼쪽으로 방향을 잡기 위해 오른쪽 지점을 leftLimit으로 설정함
      lowerRight = leftLimit
  elif (upperRight != 640 and lowerRight != 640 and upperLeft == 0 and lowerLeft == 0) : # 2개 감지 오른쪽 상단과 오른쪽 하단 감지, 설명은 윗부분과 동일
    mid = (upperRight + lowerRight) // 2
    if (640 - mid) < (mid - rightLimit) :                                                # 차선이 오른쪽으로 너무 치우쳐져 있을 때
      upperLeft = rightLimit
      lowerLeft = rightLimit

  return upperLeft, upperRight, lowerLeft, lowerRight

#=================================================================
# 직선 구간인지 판단하는 함수
# 지평선 지점 값만으로는 불확실하기 때문에 만듬
# 2차원 상자를 만들어서 반복문으로 2차원 상자를 옮기며 2차원 상자 안에 있는 이미지 배열 요소 값의 합을 구함(값이 클수록 선일 확률이 큼)
# 선에 해당하는 부분일 가능성이 높은 지점의 x축 방향 좌표를 배열에 저장함
# 배열의 표준 편차를 구해서 표준 편차가 일정 기준 작으면 직선으로 판단
#=================================================================
def judgeStraightLine(img, horizonY) :
  boxSize = 40                                  # 2차원 상자 영역 크기는 40으로 설정
  adjust = 0 if (horizonY < 265) else 40        # 계산과 오차를 줄이기 위한 조치로 직선일 가능성이 있다면 adjust 값을 40으로 함
  maxError = 50                                 # 표준 편차에서 직선인지 판단하기 위한 기준 
  
  candiLeft = []                                # 선에 해당하는 부분일 가능성이 있는 후보군을 저장하는 배열(왼쪽)
  candiRight = []                               # 선에 해당하는 부분일 가능성이 있는 후보군을 저장하는 배열(오른쪽)
  leftLimit = CENTER - adjust                   # 이미지 왼쪽만 slicing하기 위한 기준 adjust값을 통해 직선일 가능성이 있다면 계산을 줄일 수 있음, 그 이유는 slicing 된 정보가 줄기 때문
  rightLimit = CENTER + adjust                  # 이미지 오른쪽만 슬라이싱하기 위한 기준
  
  startY = int(np.floor(HEIGHT*0.25 / boxSize)) # 2차원 상자 영역을 움직이면서 각 지점마다 영역에 해당하는 요소 값의 합을 구하는데, 출발 지점 index 값
  
  leftImg = img[:, :CENTER]                     # 2차원 배열, 왼쪽 이미지
  rightImg = img[:, CENTER:][:, ::-1]           # 2차원 배열, 오른쪽 이미지, 이미지 배열을 y축을 기준으로 뒤집음
  
  for y in range(startY, 480, boxSize) :        # 왼쪽 이미지부터 2차원 상자 영역을 움직이면서 영역에 해당하는 지점의 요소들의 합을 구함
    maxi = 0                                    # 최댓값을 구하기 위한 처음 시작 default 값
    xPt = 0                                     # 선일 확률이 있는 점을 저장하는 변수: default 값은 0
    for x in range(0, leftLimit, boxSize) :     # 왼쪽에서 오른쪽으로 상자 영역이 움직임
      left = x                                  # 영역 왼쪽
      right = x + boxSize                       # 영역 오른쪽
      top = y                                   # 영역 위
      bottom = y + boxSize                      # 영역 아래
      
      total = np.sum(leftImg[top:bottom, left:right])  # 상자 영역에 해당하는 이미지 요소 값들의 합
      if maxi < total :                         # 반복문이 진행되는 동안 더 큰 값이 나올 때마다 최댓값을 업데이트함
        maxi = total
        xPt = x + (boxSize//2)                  # 선일 확률이 있는 지점의 x좌표 값은 상자 영역의 중심임
      
    candiLeft.append(xPt)                       # 후보 군에 저장
    
  isLeftNormal = (np.std(candiLeft) < maxError) # 후보 군들에서 표준편차를 구해 특정 기준 보다 아래면 True , 위면 False를 저장함

  # 아래는 오른쪽 부분이며 위 왼쪽 부분의 설명과 동일함
  for y in range(startY, 480, boxSize) :
    maxi = 0
    xPt = 640                                   # 선일 확률이 있는 점을 저장하는 변수: default 값은 640, 오른쪽이기 때문에
    for x in range(0, rightLimit, boxSize) :
      left = x
      right = x + boxSize
      top = y
      bottom = y + boxSize
      
      total = np.sum(rightImg[top:bottom, left:right])
      if maxi < total :
        maxi = total
        xPt = x + (boxSize//2)
      
    candiRight.append(xPt)
  
  isRightNormal = (np.std(candiRight) < maxError)
  
  return (isLeftNormal or isRightNormal)        # 둘 개의 변수 중 하나라도 직선이라고 판단되면 직선 구간임 

#=================================================================
# 이미지에서 지평선에 해당하는 지점의 Y좌표를 구하는 함수
# 이미지를 관찰하면 지평선을 기준으로 밝은 부분과 어두운 부분(도로, 지면)으로 나눠짐
# 밝은 부분에서 어두운 부분으로 바뀌는 지점을 찾으면 그 지점이 지평선에 해당함
# 지평선을 구하는 이유: 곡선 주행 때와 직선 주행 때의 지평선 지점 값이 달라짐
# 곡선 주행 때는 y축 방향으로 이미지의 위에서 아래로 index값이 커진다고 했을 때 지평선 지점 값이 큼
# 직선 주행 때는 곡선 주행보다 상대적으로 지평선 지점 값이 작음
# 이 차이를 이용해 곡선 주행과 직선 주행을 구분하는 데에 활용할 수 있음
# 그리고 필요한 정보는 지평선 아래, 즉 도로에 대한 정보이므로 이미지에서 확인할 부분을 최소화하는 데에 사용할 수 있음
#=================================================================
def getHorizonY(img) :
  y = 200                       # y에 할당되는 기본 값 200, y는 이미지에서 y축 방향 index 값을 의미함 
  while(img[y][CENTER] != 0) :  # width 좌표(이미지에서 x축 방향 좌표)가 CENTER(=320)인 지점에서 y값을 증가시키며 
    y += 1                      # 흰색에서 검은색으로 변하는 지점을 찾음 (이유: 인수인 img에 담긴 값은 0(=검은색) 또는 255(=흰색)로 되어있기 때문에 이런 알고리즘을 적용하는 것이 가능함)
    
  return y

#=================================================================
# 차량의 각도를 구하는 함수
# 차선의 위치 좌표일 확률이 높은 4개의 지점값을 인자로 받아 차량 각도를 구함
# 이미지의 중심 320 과 4개 점 값으로 구한 값과의 차이와 높이 값을 구해서 둘 사이의 기울기를 구함
# 이미지의 중심 320 과 4개 점 값으로 구한 값과의 차이(=x 변화량), 높이 값(=y 변화량)
# 기울기면 y/x 지만 직선 방향 값은 0이라고 치면 y/x는 무한대에 수렴함, 따라서 x/y 로 기울기의 x, y값을 서로 바꿔서
# 직선 방향값 0이 나오도록 만듬
#=================================================================
def getAngleCalculated(upperLeft, upperRight, lowerLeft, lowerRight, horizonY) :
    adjustY = horizonY + 30                                      # 기울기에서 높이를 정하기 위한 기준, 30은 임의의 값
    upperPoint = (upperLeft + upperRight) // 2                   # 가로줄1번에서 왼쪽 지점과 오른쪽 지점의 중심
    lowerPoint = (lowerLeft + lowerRight) // 2                   # 가로줄2번에서 왼쪽 지점과 오른쪽 지점의 중심
    
    meanPoint = (upperPoint + lowerPoint) // 2                   # 위에서 구한 중심들의 평균
    angle = ((meanPoint - CENTER) / float(640 - adjustY)) * 60   # x/y 값에서 임의의 값 60을 곱한 값, 임의의 값에 따라 각도 틀어짐이 달라짐
    
    # angle 값이 최대와 최소를 넘지 않도록 만들기 위한 처리
    if angle > 50 : angle = 50
    elif angle < -50 : angle = -50
    
    return int(np.round(angle)), meanPoint, adjustY              # angle 값을 반올림하여 정수형으로 반환, 중심들의 평균과 기울기에서 높이 값은 debug를 위해 필요하기 때문에 반환하는 것

#=================================================================
# 차량의 속도를 결정하는 함수
# 인자: 지평선 지점 값과 주행 방향 각도
# 직선 구간이고 차량을 주행하는 방향이 크게 틀어지지 않았을 때 속도를 높임
#=================================================================
def getSpeed(isStraightLine, angle) :
  speed = 20                                                # 기본 주행 속도는 20
  if isStraightLine < 270 and (-2 < angle and angle < 2) :  # 직선 구간일 때(지평선 지점 y축 index값이 270보다 작음), 주행 방향이 직선일 때 
    speed = 40                                              # 속도를 30으로 높임
  
  return speed

#=============================================
# 실질적인 메인 함수
# 카메라 토픽을 받아 각종 영상처리와 알고리즘을 통해
# 차선의 위치를 파악한 후에 조향각을 결정하고,
# 최종적으로 모터 토픽을 발행하는 일을 수행함.
#=============================================
def start():

    # 위에서 선언한 변수를 start() 안에서 사용하고자 함
    global motor, image

    #=========================================
    # ROS 노드를 생성하고 초기화 함.
    # 카메라 토픽을 구독하고 모터 토픽을 발행할 것임을 선언
    #=========================================
    rospy.init_node('driving')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw/",Image, img_callback)

    print ("----- Xycar self driving -----")

    # 첫번째 카메라 토픽이 도착할 때까지 기다림.
    rospy.wait_for_message("/usb_cam/image_raw/", Image)

    #=========================================
    # 메인 루프
    # 카메라 토픽이 도착하는 주기에 맞춰 한번씩 루프를 돌면서
    # "이미지처리 + 차선위치찾기 + 조향각 결정 + 모터토픽 발행"
    # 작업을 반복적으로 수행함.
    #=========================================
    while not rospy.is_shutdown():

        # 이미지 처리를 위해 카메라 원본 이미지를 img에 복사 저장한다.
        img = image.copy()
        
        #=========================================
        # 차선을 정확하게 인식하기 위한 이미지 처리
        #=========================================
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                            # 연산 적게 하기 위해서 복사한 원본 이미지 img를 gray scale 이미지로 변환함
        ret, blackAndWhiteImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # gray scale로 변환된 이미지의 RGB값에서 임계값(=0) 이상인 것은 흰색(=255)로 변환
        # cv2.imshow('bw', blackAndWhiteImg)

        # 지평선 지점 y 값을 구함
        horizonY = getHorizonY(blackAndWhiteImg)                                                   # 0과 255, 2개의 값으로 구성된 이미지(blackAndWhiteImg)를 인자로 넣어서 지평선 지점 y좌표(index 값)을 horizonY 변수에 할당함 
        
        # 원본 이미지를 위에서 내려다보는 듯한 이미지로 변환
        wrapped_img = wrapping(blackAndWhiteImg)                                                   # blackAndWhiteImg가 인자로 들어감
        # cv2.imshow('wrapped', wrapped_img)
        
        # 직선 구간 인지 판단함
        isStraightLine = judgeStraightLine(wrapped_img, horizonY)                                  # isStraightLine에는 True / False 데이터가 할당됨, 직선 구간이면 True이고 아니면 False임

        # 관심 영역을 설정함
        # roi_img = roi(wrapped_img)                                                               # 변환된 이미지가 인자임, 차선 인식에 방해되는 주변의 흰색 부분은 신경 쓰지 않기 위해서 이미지에서 관심 영역을 설정함
        # cv2.imshow('roi', roi_img)

        # 이미지 내에서 차선일 확률이 높은 위치를 찾음 
        upperLeft, upperRight, lowerLeft, lowerRight = getLineX(wrapped_img, horizonY, isStraightLine)  # 변환된 이미지와 지평선 y값이 인자로 들어감
        
        #=========================================
        # 핸들 조향각 값인 angle값 정하기.
        # 차선의 위치 정보를 이용해서 angle값을 설정함.
        #=========================================
        angle, meanPoint, slopeY = getAngleCalculated(upperLeft, upperRight, lowerLeft, lowerRight, horizonY)
        
        #=========================================
        # 차량의 속도 값인 speed값 정하기.
        # 주행 속도를 조절하기 위해 speed값을 설정함.
        #=========================================
        speed = getSpeed(isStraightLine, angle)

        # debug를 위해서 이미지에 차량의 주행 방향을 시각화 (화살표로 나타냄)
        cv2.arrowedLine(img, (320, 480), (meanPoint, (HEIGHT-slopeY)), (0, 0, 255), 2)
        cv2.imshow('result', img)
        cv2.waitKey(1)

        # drive() 호출. drive()함수 안에서 모터 토픽이 발행됨.
        drive(angle, speed)


#=============================================
# 메인 함수
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함
# start() 함수가 실질적인 메인 함수임.
#=============================================
if __name__ == '__main__':
    start()

