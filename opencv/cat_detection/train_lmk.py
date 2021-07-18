import datetime

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import Input
import numpy as np

img_size = 224

mode = 'lmks' # [bbs, lmks]
if mode is 'bbs':
  output_size = 4
elif mode is 'lmks':
  output_size = 18

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# 에러 발생 // ValueError: Object arrays cannot be loaded when allow_pickle=False
# 먼저 기존의 np.load를 np_load_old에 저장해둠.
np_load_old = np.load
# 기존의 parameter을 바꿔줌
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

print('dataloads start!')

data_00 = np.load('dataset/CAT_00.npy')
data_01 = np.load('dataset/CAT_01.npy')
data_02 = np.load('dataset/CAT_02.npy')
data_03 = np.load('dataset/CAT_03.npy')
data_04 = np.load('dataset/CAT_04.npy')
data_05 = np.load('dataset/CAT_05.npy')
data_06 = np.load('dataset/CAT_06.npy')


np_load_old = np.load
# 기존의 parameter을 바꿔줌
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

print('dataloads start!')

data_00 = np.load('dataset/CAT_00.npy')
data_01 = np.load('dataset/CAT_01.npy')
data_02 = np.load('dataset/CAT_02.npy')
data_03 = np.load('dataset/CAT_03.npy')
data_04 = np.load('dataset/CAT_04.npy')
data_05 = np.load('dataset/CAT_05.npy')
data_06 = np.load('dataset/CAT_06.npy')

print('dataloads finish!')
print('data preprocessing start!')

x_train = np.concatenate((data_00.item().get('imgs'), data_01.item().get('imgs'), data_02.item().get('imgs'), data_03.item().get('imgs'), data_04.item().get('imgs'), data_05.item().get('imgs')), axis=0)
y_train = np.concatenate((data_00.item().get(mode), data_01.item().get(mode), data_02.item().get(mode), data_03.item().get(mode), data_04.item().get(mode), data_05.item().get(mode)), axis=0)
x_test = np.array(data_06.item().get('imgs'))
y_test = np.array(data_06.item().get(mode))

# 이미지를 0~1로 바꿔줌  (메모리 ㅈ버그걸림)
x_train = x_train.astype('float64') / 255.
x_test = x_test.astype('float64') / 255.
x_train = np.reshape(x_train, (-1, img_size, img_size, 3))
x_test = np.reshape(x_test, (-1, img_size, img_size, 3))

y_train = np.reshape(y_train, (-1, output_size))
y_test = np.reshape(y_test, (-1, output_size))

inputs = Input(shape=(img_size, img_size, 3))


mobilenetv2_model = mobilenet_v2.MobileNetV2(input_shape=(img_size, img_size, 3), alpha=1.0, include_top=False, weights='imagenet', input_tensor=inputs, pooling='max')

net = Dense(128, activation='relu')(mobilenetv2_model.layers[-1].output)
net = Dense(64, activation='relu')(net)
net = Dense(output_size, activation='linear')(net)