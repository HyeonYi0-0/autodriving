import cv2
import numpy as np

CENTER = 320
ADJUSTVAL = 10
temp = 100

def roi(image):
    shape = np.array([[(int(640*0), int(480*0.2)),
                       (int(640*1), int(480*0.2)),
                       (int(640*1), int(480*1)),
                       (int(640*0), int(480*1))]])

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, shape, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def wrapping(image):
    (h, w) = (image.shape[0], image.shape[1]) #h: 480, w: 640

    #아래 상수 바꿀 필요성
    # [x,y] 좌표점을 4x2의 행렬로 작성
    # 좌표점은 좌상->우상->좌하->우하
    before= [[int(640*0.3), int(480*0.65)], [int(640*0.7), int(480*0.65)],
             [int(640*0.05), int(480*0.85)], [int(640*0.95), int(480*0.85)]]
    after = [[int(640*0.25), int(480*0.2)], [int(640*0.75), int(480*0.2)],
             [int(640*0.25), int(480*0.85)], [int(640*0.75), int(480*0.85)]]
    
    source = np.float32(before)
    destination = np.float32(after)

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    _image = cv2.warpPerspective(image, transform_matrix, (w, h))
    
    return _image

# lineThickness = 20
# def getTrueLeftLine(inner, out) :
#   criteria = CENTER // 2
#   lineX = out
  
#   if (inner-out > lineThickness) and (np.abs(criteria-inner) < np.abs(criteria-out)):
#     lineX = inner
    
#   return lineX

# def getTrueRightLine(inner, out) :
#   criteria = (CENTER // 2) * 3
#   lineX = out
  
#   if (out-inner > lineThickness) and (np.abs(criteria-inner) < np.abs(criteria-out)):
#     lineX = inner
    
#   return lineX

# def getLineX(img, horizonY) :
#   bottom2horizon = 480 - horizonY
#   upper = int(horizonY + np.round(bottom2horizon*0.1))
#   lower = int(horizonY + np.round(bottom2horizon*0.9))

#   upperImgArr = img[upper]
#   lowerImgArr = img[lower]
  
#   out = np.argmax(upperImgArr[:CENTER])
#   inner = CENTER - np.argmax(upperImgArr[:CENTER][::-1])
#   upperLeft = getTrueLeftLine(inner, out)
  
#   out = 640 - np.argmax(upperImgArr[CENTER:][::-1])
#   inner = CENTER + np.argmax(upperImgArr[CENTER:])
#   upperRight = getTrueRightLine(inner, out)
  
#   out = np.argmax(lowerImgArr[:CENTER])
#   inner = CENTER - np.argmax(lowerImgArr[:CENTER][::-1])
#   lowerLeft = getTrueLeftLine(inner, out)
  
#   out = 640 - np.argmax(lowerImgArr[CENTER:][::-1])
#   inner = CENTER + np.argmax(lowerImgArr[CENTER:])
#   lowerRight = getTrueRightLine(inner, out)
  
#   upLen = upperRight - upperLeft
#   lowLen = lowerRight - lowerLeft
  
#   if (upperLeft == 0 and upperRight != 640 and lowerLeft != 0 and lowerRight != 640) : # 3개 감지 왼쪽 상단 감지 X
#     upperLeft = upperRight - lowLen
#   elif (upperLeft != 0 and upperRight == 640 and lowerLeft != 0 and lowerRight != 640) : # 3개 감지 오른쪽 상단 감지 X
#     upperRight = upperLeft + lowLen
#   elif (upperLeft != 0 and upperRight != 640 and lowerLeft == 0 and lowerRight != 640) : # 3개 감지 왼쪽 하단 감지 X
#     lowerLeft = lowerRight - upLen
#   elif (upperLeft != 0 and upperRight != 640 and lowerLeft != 0 and lowerRight == 640) : # 3개 감지 오른쪽 하단 감지 X
#     lowerRight = lowerLeft + upLen
#   elif (upperLeft != 0 and lowerLeft != 0 and upperRight == 640 and lowerRight == 640) :
#     mid = (upperLeft + lowerLeft) // 2
#     direc = int(np.abs(upperLeft-lowerLeft) / float(upper-lower))
#     if mid < (CENTER-mid) : # 왼쪽으로 너무 취우쳐져 있을 때
#       upperRight = CENTER
#       lowerRight = CENTER
#     elif -3 < direc and direc < 3 : # 직선 구간일 때
#       upperRight = 640 - upperLeft
#       lowerRight = 640 - lowerLeft
#   elif (upperRight != 640 and lowerRight != 640 and upperLeft == 0 and lowerLeft == 0) :
#     mid = (upperRight + lowerRight) // 2
#     direc = int(np.abs(upperRight-lowerRight) / float(upper-lower))
#     if (640 - mid) < (mid - CENTER) : # 오른쪽으로 너무 취우쳐져 있을 때
#       upperLeft = CENTER
#       lowerLeft = CENTER
#     elif -3 < direc and direc < 3  :
#       upperLeft = 640 - upperRight
#       lowerLeft = 640 - lowerRight

#   return upperLeft, upperRight, lowerLeft, lowerRight

# sliding window를 만들어서 한다 반반씩
# 일단 분산을 구해서 특정 분산값에 따라 비정상인지 정상인지 판단한다.
# 만약 비정상이라면 한쪽 라인에 따라 방향을 설정한다
# 그리고 직선 구간에서는 지평선의 높이가 높을 때는 반반 Sliding 할 때 그 정도를 조절한다.
def getLineX(img, horizonY) :
  debug_img = np.dstack((img, img, img))
  
  boxSize = 40 # 480 * 640 ==>20:  24 * 32 // 40:  12 * 16 // 480 * 0.2 = 96 
  predleftX = 0
  predrightX = 640
  adjust = 0 if (horizonY < 265) else 40
  
  candiLeft = []
  candiRight = []
  leftLimit = CENTER - adjust
  rightLimit = CENTER + adjust
  
  # startY = (boxSize // 2) + boxSize * (int(np.ceil(96 / boxSize)) - 1)
  # startX = (boxSize // 2)
  
  startY = int(np.floor(96 / boxSize))
  
  leftImg = img[:, :CENTER]
  rightImg = img[:, CENTER:][:, ::-1]
  # cv2.imshow("hi", rightImg)
  
  # 왼쪽 먼저 해보자
  for y in range(startY, 480, boxSize) :
    maxi = 0
    xPt = 0
    for x in range(0, leftLimit, boxSize) :
      left = x
      right = x + boxSize
      top = y
      bottom = y + boxSize
      
      total = np.sum(leftImg[top:bottom, left:right])
      if maxi < total :
        maxi = total
        xPt = x + (boxSize//2)
      
    candiRight.append(xPt)
  
  predleftX = int(np.mean(candiRight))
  isLeftNormal = np.std(candiRight)
  # print(predleftX, isLeftNormal)

  # 이제 오른쪽 해보자
  for y in range(startY, 480, boxSize) :
    maxi = 0
    xPt = 640
    for x in range(0, rightLimit, boxSize) :
      left = x
      right = x + boxSize
      top = y
      bottom = y + boxSize
      
      total = np.sum(rightImg[top:bottom, left:right])
      if maxi < total :
        maxi = total
        xPt = x + (boxSize//2)
      
    candiLeft.append(xPt)
  
  predrightX = 640 - int(np.mean(candiLeft))
  isRightNormal = np.std(candiLeft)
  # print(predrightX, isRightNormal)
  
  cv2.line(debug_img, (predleftX, 240), (predrightX, 240), [0, 0, 255], 2)
  cv2.imshow('debug', debug_img)
  
  # 비정상인 경우를 찾자
  # 한쪽이 비정상이고 다른 한쪽이 정상일 때
  # 정상인 쪽을 바탕으로 스티어링 
  # 둘다 정상이긴 한데 한쪽으로 너무 붙었을 때 
  if predleftX < (CENTER-predleftX) : # 왼쪽으로 너무 취우쳐져 있을 때
    predrightX = CENTER
  elif (640-predrightX) < (predrightX-CENTER) : # 오른쪽으로 너무 취우쳐져 있을 때
    predleftX = CENTER
  elif isLeftNormal >= 60 and isRightNormal < 60:
    if isRightNormal < 20 :                    # 직선일 때
      predleftX = 640 - predrightX
    else :                                     # 곡선일 때
      predleftX = 0
  elif isLeftNormal < 60 and isRightNormal >= 60:
    if isLeftNormal < 20 :                     # 직선일 때
      predrightX = 640 - predleftX
    else :                                     # 곡선일 때
      predrightX = 640
  
  return predleftX, predrightX
  

def getHorizonY(img) :
  ## img는 w: 640, h: 480의 2차원 배열(480 X 640)
  y = 200                    # 기본 값
  while(img[y][320] != 0) :  # width 좌표가 320인 지점에서 y값을 증가시키며 
    y += 1                   # 흰색에서 검은색으로 변하는 지점을 찾음
    
  return y

# height은 지평선에 의해 조절됨
def getAngleCalculated(left, right) :
    meanPoint = (left + right) // 2
    angle = ((meanPoint - CENTER) / float(200)) * 50
    
    if angle > 50 : angle = 50
    elif angle < -50 : angle = -50
    
    return int(np.round(angle)), meanPoint
  
def getSpeed(horizonY, angle) :
  speed = 22
  if horizonY < 270 and (-2 < angle and angle < 2) :
    speed = 50
  
  return speed

#=============================
# main
#=============================
'''
step1
  line detection:
    현재 인터넷에서 찾은 방식을 그대로 적용(HoughP() 함수보다 처리 속도가 더 빠름 + 곡선 처리가 더 능숙)
    일단 인터넷에서 찾은 방식 분석
    보완 요소:
      과제에서는 차선이 흰색이므로 흰색만 추출하는 필터가 필요할 듯 ==> 이진화로 처리
      ROI영역 개선 필요, 이유는 도로에 있는 횡단보도 및 여러 표시들이 detection 되기 때문
      ==> midpoint를 기준으로 왼쪽 ~ mid에서 처음으로 흰색 부분이 나타나는 지점, 오른쪽에서 ~ mid에서 처음으로 흰색 나타나는 지점
      ROI영역 직선일 때는 멀리 보고, 곡선일 때는 가까이 보자(지평선을 찾는 알고리즘이 필요)
      ==> width 기준이 midpoint면 height 기준은 지평선
      detection이 되지 않을 때:
        한쪽이 안되는 경우:
          수평선이 낮을 때(코너링 구간) horizonY >= 270:
            디폴트 값 설정으로 해결
          
          수평선이 높을 때(직선 구간) horizonY < 270:
            한쪽으로 너무 치우쳐져 있으면 mid 값과 감지된 한쪽 부분의 x좌표를 디폴트로 하자
            
        양쪽이 안되는 경우(디폴트 값으로 0과 640이 설정될 때):
          수평선이 낮을 때(코너링 구간):
            기존의 horizontal line을 내린다. (한쪽이라도 감지될 때까지)
            근데 감지가 되지 않는다. 
          
          수평선이 높을 때(직선 구간):
            기존의 horizontal line을 내린다. (한쪽이라도 감지될 때까지)
            근데 감지가 되지 않는다. (i == 480이다.) 그러면 0과 640으로 함.
      
step2
  steering: 일단 완료
    자동차의 중심(width // 2) 점과 라인에서 detection된 두 점 사이의 중심점과의 기울기에 따라 angle 적용
    이때, x, y축 y = x에 대칭시켜야 함, 이유는 기울기가 작아질 수록 각도를 더 틀어야하기 때문

step3 
  speed:
    계획은 직선 구간에서 속도를 높이고 곡선 구간에서는 속도를 줄일 예정
    가능한 선까지 속도를 높이는 방향
    직선인지 아는 것은 steering에서 기울기가 수직에 가까울 수록(steering=0)일 때 속도를 높임
    직선 구간에는 일반 주행과 고속 주행을 구분함 기준은 steering 각도
    곡선 구간에는 일반 주행보다 더 속도를 줄이거나 그대로 유지하거나(이는 실제로 주행해보며 조정)
'''
img = cv2.imread("../img0.png")

'''
beforeArr = [[int(640*0.3), int(480*0.65)], [int(640*0.7), int(480*0.65)],
            [int(640*0.05), int(480*0.85)], [int(640*0.95), int(480*0.85)]]
afterArr = [[int(640*0.25), int(480*0.2)], [int(640*0.75), int(480*0.2)],
            [int(640*0.25), int(480*0.85)], [int(640*0.75), int(480*0.85)]]
'''
#debuging code
#cv2.line(img, beforeArr[0], beforeArr[1], (0, 255, 255), 2)
#cv2.line(img, beforeArr[2], beforeArr[3], (255, 0, 255), 4)

#cv2.line(img, afterArr[0], afterArr[1], (0, 255, 255), 2)
#cv2.line(img, afterArr[2], afterArr[3], (0, 0, 255), 2)

grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, blackAndWhiteImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# 지평선 구하기
horizonY = getHorizonY(blackAndWhiteImg)
# cv2.imshow('bw', blackAndWhiteImg)
  
## 조감도 wrapped img
wrapped_img = wrapping(blackAndWhiteImg)
# cv2.imshow('wrapped', wrapped_img)

# getLineX(wrapped_img, horizonY)

# ##조감도 필터링 자르기
roi_img = roi(wrapped_img)
# cv2.imshow('roi', roi_img)

# ## 선 찾기
left, right = getLineX(roi_img, horizonY)
# print(left, right)

angle, meanPoint= getAngleCalculated(left, right)
# print(angle)

speed = getSpeed(horizonY, angle)

cv2.arrowedLine(img, (320, 480), (meanPoint, (300)), (0, 0, 255), 2)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()