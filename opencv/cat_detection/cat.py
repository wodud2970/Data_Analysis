import random
import dlib, cv2, os
import pandas as pd
import numpy as np

dirname ="CAT_00"
base_path = "archive/%s" % dirname
file_list = sorted(os.listdir(base_path)) #path의 경로를 리스트로 만들고 정렬

for f in file_list:
    #file_list에 cat이 없으면
    if '.cat' not in f:
        continue

    #read landmarks
    pd_frame = pd.read_csv(os.path.join(base_path, f), sep = ' ', header = None)
    # [1:-1] 1는 점의 갯수 나머지 9개를 2개의 열로 int type으로 나누어준다
    # landmarks = (pd.frame.values()[0][1:-1]).reshape((-1,2)).astype(np.int)
    landmarks = (pd_frame[range(1,19)].values).reshape((-1, 2)).astype(np.int)
    #load lnadmakrs (cv2로 불러온다)
    img_filename, ext = os.path.splitext(f)
    img = cv2.imread(os.path.join(base_path, img_filename))

    #visualize
    for l in landmarks:
        cv2.circle(img, center=tuple(l), radius=1, color = (0,0,255), thickness=2)

    cv2.imshow('cat_dot',img)
    if cv2.waitKey(0) == ord('q'):
        break

#1. Face Detection

#2. Face Landmark Detection