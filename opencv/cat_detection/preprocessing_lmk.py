import random

import dlib, cv2, os
import pandas as pd
import numpy as np

img_size = 224
# dirname ="CAT_00"
# base_path = "archive/%s" % dirname
# file_list = sorted(os.listdir(base_path)) #path의 경로를 리스트로 만들고 정렬
# random.shuffle(file_list)


#데이터 셋 저장
dataset = {
    'imgs' : [],
    'lmk' : [],
    'bbs' : []
}

# 사진의 공백을 검은색으로 채워서 정사각형을 만들어준다
def resize_img(im):
    old_size = im.shape[:2] #old_size is in (height, width) format
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    #new_size should be in (width, height) format ..상하를 바꾸어준다
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h //2, delta_h - (delta_h // 2)
    left, right = delta_w //2, delta_w - (delta_w // 2)
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value = [0,0,0])
    return new_im, ratio, top, left

for i in range(1,7):
    dirname = "CAT_0" + "%s" % i
    base_path = "archive/%s" % dirname
    file_list = sorted(os.listdir(base_path))  # path의 경로를 리스트로 만들고 정렬
    random.shuffle(file_list)
    for f in file_list:
        if '.cat' not in f:
            continue
        pd_frame = pd.read_csv(os.path.join(base_path, f), sep = ' ', header = None)
        landmarks = (pd_frame[range(1,19)].values).reshape((-1, 2)).astype(np.int)
        bb =np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)]).astype(np.int)
        center = np.mean(bb, axis=0)
        face_size = max(np.abs(np.max(landmarks, axis = 0) - np.min(landmarks, axis=0)))
        #bb가 타히트가 주기 때문에 느쓴하게 bb를 만들어준다
        new_bb = np.array([
            center - face_size * 0.6,
            center - face_size * 0.6
        ]).astype(np.int)
        #마이너스 값을 없애도록 조정 (np.clip() : 행렬안의 값을 min, max로 제한 )
        new_bb = np.clip(new_bb, 0 , 99999)
        #좌표값을 조정해주기 위해서 즉 (0,0) 위치를 조정해주기 위해
        new_landmarks = landmarks - new_bb[0]

        #load lnadmakrs (cv2로 불러온다)
        img_filename, ext = os.path.splitext(f)
        img = cv2.imread(os.path.join(base_path, img_filename))

        new_img = img[new_bb[0][1] : new_bb[1][1], new_bb[0][0], new_bb[1][0]]

        #resize image 와 relocate landmarks
        img, ratio, top, left = resize_img(img)
        # 리사이즈를 통해 변화 랜드마크 위치를 바꾸어준다
        # landmarks = ((landmarks * ratio) + np.array([left,top])).astype(np.int)
        new_landmarks = ((new_landmarks * ratio) + np.array([left,top])).astype(np.int)

        #Bounding box
        bb = np.array([np.min(landmarks, axis = 0), np.max(landmarks, axis = 0)])

        dataset['imgs'].append(img)
        dataset['lmk'].append(new_landmarks.flatten())
        dataset['bbs'].append(new_bb.flatten())

    #완성된 데이터 셋 저장
    np.save('dataset/lmks%s.npy' % dirname, np.array(dataset))