#dlib 설치방법 개허무하다
#conda install -c anaconda cmake
#conda install -c conda-forge dlib

import cv2, dlib, sys
import numpy as np

cap = cv2.VideoCapture('girl.mp4')

#이미지 씌우기
overlay = cv2.imread('smile.png', cv2.IMREAD_UNCHANGED)

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    try:
        bg_img = background_img.copy()
        # convert 3 channels to 4 channels
        if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

        b, g, r, a = cv2.split(img_to_overlay_t)

        mask = cv2.medianBlur(a, 5)

        h, w, _ = img_to_overlay_t.shape
        roi = bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

        bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img1_bg, img2_fg)

        # convert 4 channels to 4 channels
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
        return bg_img
    except Exception:
        return background_img

scaler = 0.3

#학습된 모델
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')



while True:
    ret, img = cap.read()
    if not ret:
        break
    #img 사이즈를 줄인다
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    ori = img.copy()

    #얼굴 찾기
    faces = detector(img)
    face = faces[0]

    #얼굴 특징점 찾기 (모델을 이용하여 ) 눈, 코, 입을 찾는다
    dlib_shape = predictor(img, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    #왼쪽위 , 오른쪽아래
    top_left = np.min(shape_2d, axis = 0)
    bottom_right = np.max(shape_2d, axis = 0)

    #얼굴크기(1.8은 가면이 얼굴보다 커야하기 때문에 ) int로 좌표를찍지않으면 값이 큰가보네
    face_size = int(max(bottom_right - top_left) *1.8)

    #얼굴의 중심점 찾기
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

    #얼굴 좌표를 계산하여 overlay를 씌어준다
    result = overlay_transparent(ori, overlay, center_x, center_y, overlay_size=(face_size, face_size))

    #사각형 visularize
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255,255,255,255),
                        thickness = 2, lineType = cv2.LINE_AA)
    #얼굴 점 찍기
    for s in shape_2d:
        cv2.circle(img, center = tuple(s), radius=1, color=(255,255,255),thickness=2, lineType=cv2.LINE_AA)
    #왼쪽위와 오른쪽 아래 파랜색 점
    cv2.circle(img, center=tuple(top_left), radius=1, color = (255,0,0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(bottom_right), radius=1, color = (255,0,0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color = (0,0,255), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.imshow('result', result)
    #멈추는 key
    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)