import cv2
import numpy as np

CENTER = 320
ADJUSTVAL = 15

def roi(image):
    shape = np.array([[(int(640*0), int(480*0)),
                       (int(640*1), int(480*0)),
                       (int(640*0.75), int(480*1)),
                       (int(640*0.25), int(480*1))]])

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

def getLineX(img, horizonY) :
    adjustY = horizonY + ADJUSTVAL
    horizonLineArr = img[adjustY]
    leftlimit = CENTER - 10
    rightlimit = CENTER + 10
    leftX = np.argmax(horizonLineArr[:leftlimit])
    rightX = 640 - np.argmax(horizonLineArr[rightlimit:][::-1])
    
    return leftX, rightX

def getHorizonY(img) :
  ## img는 w: 640, h: 480의 2차원 배열(480 X 640)
  y = 150                    # 기본 값
  while(img[y][320] != 0) :  # width 좌표가 320인 지점에서 y값을 증가시키며 
    y += 1                   # 흰색에서 검은색으로 변하는 지점을 찾음
    
  return y

# height은 지평선에 의해 조절됨
def getAngleCalculated(left, right, height) :
    adjustY = height + ADJUSTVAL
    point = (left + right) // 2
    angle = (point - CENTER) / float(640-adjustY)
    
    return int(np.ceil(angle))

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
        직선 구간 && 곡선 구간에서 차선이 끊긴 경우:
          최대한 추출된 x좌표에서 steering
          그리고 이전에 주행하던 각도 참고해서 방향 설정 보완
        
        방지턱에서 차선이 감지되지 않는 경우:
          이전 주행하던 각도와 스피드로 방향 설정 보완
          아니면 default 주행 영역을 설정
      
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
img = cv2.imread("img4.png")

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

# ##조감도 필터링 자르기
roi_img = roi(wrapped_img)
# cv2.imshow('roi', roi_img)

# ## 선 찾기
left, right = getLineX(roi_img, horizonY)
# print(left, right)

angle = getAngleCalculated(left, right, horizonY)
# print(angle)

cv2.line(img, ((left+right) // 2, horizonY+ADJUSTVAL), (320, 480), (0, 255, 255), 2)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()