import cv2
import numpy as np

video_path = 'redvelvet.mp4'
cap = cv2.VideoCapture(video_path)

output_size = (375, 667) #width height

# initialize writeing video
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
#[파일이름,코덱, 프레임,사이즈]
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]),fourcc, cap.get(cv2.CAP_PROP_FPS),output_size)

if not cap.isOpened():
    exit()
#속도와 정확도가 적당히 나오는 Tracker (Model)
#Tracker 모델을 더 잘 이용하면 좋게 사용할수 있다
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "mil": cv2.TrackerMIL_create,
}
#업그레이드 버젼 변수
# global variables
top_bottom_list, left_right_list = [], []
count = 0
fit_to = 'height'

#tracker를 가져온다
tracker = cv2.TrackerCSRT_create()
#tracker = OPENCV_OBJECT_TRACKER['csrt']


#첫번째 프레임을 읽어온다
for i in range(100):
    ret, img = cap.read()
cv2.namedWindow('Select Window')
cv2.imshow('Select Window', img)


#여기를 얼굴인식 바운딩 박스처리를해주면 그거에 맞추어 카메라 옮겨질거 같다!
#setting ROI (객체를 세팅한다) space를 눌러야 세팅 완료
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select Window')

#initialize tracker
tracker.init(img, rect)



while True:
    ret, img = cap.read()

    if not ret:
        exit()

    #image 를 따라가게 만든다
    suceess, box = tracker.update(img)

    left, top, w, h = [int(v) for v in box]



    #유튜브 버젼
    center_x = left + w /2
    center_y = top + h /2

    #동영상 저장 좌표
    #좌표계산이 좀 잘못된거 같다 확인 해야할듯 수학이 좀 안되나방
    result_top = int(center_y - output_size[1]/2 - 100)
    result_bottom = int(center_y + output_size[1]/2 + 100)
    result_left = int(center_x - output_size[0]/2 +100)
    result_right = int(center_x + output_size[0]/2 - 100)

    # result_img = img[result_top:result_bottom, result_left:result_right]
    result_img = img[result_top:result_bottom, result_left:result_right].copy()

    # #업그레이드 버젼
    # right = left + w
    # bottom = top + h
    #
    # # save sizes of image
    # top_bottom_list.append(np.array([top, bottom]))
    # left_right_list.append(np.array([left, right]))
    #
    # # use recent 10 elements for crop (window_size=10)
    # if len(top_bottom_list) > 10:
    #     del top_bottom_list[0]
    #     del left_right_list[0]
    #
    # # compute moving average
    # avg_height_range = np.mean(top_bottom_list, axis=0).astype(np.int)
    # avg_width_range = np.mean(left_right_list, axis=0).astype(np.int)
    # avg_center = np.array([np.mean(avg_width_range), np.mean(avg_height_range)])  # (x, y)
    #
    # # compute scaled width and height
    # scale = 1.3
    # avg_height = (avg_height_range[1] - avg_height_range[0]) * scale
    # avg_width = (avg_width_range[1] - avg_width_range[0]) * scale
    #
    # # compute new scaled ROI
    # avg_height_range = np.array([avg_center[1] - avg_height / 2, avg_center[1] + avg_height / 2])
    # avg_width_range = np.array([avg_center[0] - avg_width / 2, avg_center[0] + avg_width / 2])
    #
    # # fit to output aspect ratio
    # if fit_to == 'width':
    #     avg_height_range = np.array([
    #         avg_center[1] - avg_width * output_size[1] / output_size[0] / 2,
    #         avg_center[1] + avg_width * output_size[1] / output_size[0] / 2
    #     ]).astype(np.int).clip(0, 9999)
    #
    #     avg_width_range = avg_width_range.astype(np.int).clip(0, 9999)
    # elif fit_to == 'height':
    #     avg_height_range = avg_height_range.astype(np.int).clip(0, 9999)
    #
    #     avg_width_range = np.array([
    #         avg_center[0] - avg_height * output_size[0] / output_size[1] / 2,
    #         avg_center[0] + avg_height * output_size[0] / output_size[1] / 2
    #     ]).astype(np.int).clip(0, 9999)
    #
    # # crop image
    # result_img = img[avg_height_range[0]:avg_height_range[1], avg_width_range[0]:avg_width_range[1]].copy()
    #
    # # resize image to output size
    # result_img = cv2.resize(result_img, output_size)
    #
    # # visualize
    # pt1 = (int(left), int(top))
    # pt2 = (int(right), int(bottom))
    # cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)



    #Bounding box
    cv2.rectangle(img, pt1 = (left, top), pt2=(left +w ,top +h), color = (255,255,255),thickness=3)

    out.write(result_img)

    cv2.imshow('img', img)
    cv2.imshow('result', result_img)
    if cv2.waitKey(1) == ord('q'):
        break
