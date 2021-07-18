import datetime

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import Input
import numpy as np

img_size = 224

mode = 'bbs' #[bbs, lmks]

if mode is 'bbs':
    output_size = 4 #(x1, y1, x2, y2)
elif mode is 'lmks':
    output_size = 18

start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

#train_data
data_00 = np.load('dataset/CAT_00.npy', allow_pickle=True)
data_01 = np.load('dataset/CAT_01.npy', allow_pickle=True)
data_02 = np.load('dataset/CAT_02.npy', allow_pickle=True)
data_03 = np.load('dataset/CAT_03.npy', allow_pickle=True)
data_04 = np.load('dataset/CAT_04.npy', allow_pickle=True)
data_05 = np.load('dataset/CAT_05.npy', allow_pickle=True)
#val_data
data_06 = np.load('dataset/CAT_06.npy', allow_pickle=True)
#이미지를 넣어준다 (np 합치기)
x_train = np.concatenate((data_00.item().get('imgs'), data_01.item().get('imgs'), data_02.item().get('imgs'), data_03.item().get('imgs')), axis=0)
#Bounding box를 y값으로 해준다 (np 합치기)
y_train = np.concatenate((data_00.item().get(mode), data_01.item().get(mode), data_02.item().get(mode), data_03.item().get(mode)), axis=0)

#validation set (cross validation 을 사용해주어도 된다)
x_test = np.array(data_06.item().get('imgs'))
y_test = np.array(data_06.item().get(mode))

#255으로 나누어주어 ( 0 ~ 1 사이의 값으로 해준다)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#이미지 사이즈를 맞추어준다
x_train = np.reshape(x_train, (-1, img_size, img_size, 3))
x_test = np.reshape(x_test, (-1, img_size, img_size, 3))
y_train = np.reshape(y_train, (-1, output_size))
y_test = np.reshape(y_test, (-1, output_size))

#Model 정의
inputs = Input(shape=(img_size, img_size, 3))

#가볍게 pretrained 된 모델
mobilenet_v2_model =mobilenet_v2.MobileNetV2(
    input_shape=(img_size, img_size, 3),
    alpha=1.0,
    include_top=False, #true로 하면 image_net을 output으로 내보낸다 (Classification) 우리꺼는 Regreesion
    weights='imagenet',
    input_tensor=inputs,
    pooling='max'
                                             )
net = Dense(128, activation='relu')(mobilenet_v2_model.layers[-1].output)
net = Dense(64, activation='relu')(mobilenet_v2_model.layers[-1].output)
net = Dense(output_size, activation='relu')(mobilenet_v2_model.layers[-1].output)
model = Model(input=inputs, outputs = net)

model.summary()

#training
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
model.fit(x_train, y_train, epochs = 50, batch_size= 32, shuffle=True, validation_data=(x_test,y_test),verbose=1,
          callbacks=[
              TensorBoard(log_dir='logs/%s' % (start_time)),
              ModelCheckpoint('./models/%s.h5' %(start_time),monitor='val_loss', verbose = 1,
                              save_best_only=True,mode='auto'),
                              ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,verbose=1, mode='auto')

          ]
)