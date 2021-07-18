import numpy as np

rows = np.loadtxt("./lotto.csv", delimiter=",")
row_count = len(rows)

#당첨번호를 원핫인코딩벡터(onbin)으로 변환
def numbers2ohbin(numbers):

    ohbin = np.zeros(45)

    for i in range(6):
        ohbin[int(numbers[i]-1)] = 1 #로또번호가 1부터 시작하지만 벡터의 인덱스는 0부터 시작하므로 1을뺌

    return ohbin

#원핫인코딩벡터(ohbin)을 번호로 변환
def ohbin2numbers(ohbin):

    numbers = []

    for i in range(len(ohbin)):
        if ohbin[i] ==1.0:
            numbers.append(i + 1)

    return numbers

#모델 구성및 학습
numbers = rows[:, 1:7]
ohbins = list(map(numbers2ohbin,numbers))

#전날
x_samples = ohbins[0:row_count-1]
#다음날
y_samples = ohbins[1:row_count]


train_idx = (0, 800)
val_idx = (801, 900)
test_idx = (901, len(x_samples))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

#모델정의

model = keras.Sequential([
    keras.layers.LSTM(128, batch_input_shape = (1,1,45), return_sequences=False, stateful= True),
    keras.layers.Dense(45, activation = "sigmoid")
])

#모델 컴파일


model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=['acc'])

#에포크마다 훈련과 검증의 손실 및 정확도 기록하기위한 변수
train_loss = []
train_acc = []
val_loss = []
val_acc = []

#epoch 100
for epoch in range(100):

    model.reset_states() #에포크마다 1회부터 다시 훈련하므로 상태 초기화
    batch_train_loss = []
    batch_train_acc = []

    for i in range(train_idx[0], train_idx[1]):
        xs = x_samples[i].reshape(1, 1, 45)
        ys = y_samples[i].reshape(1,45)

        loss, acc = model.train_on_batch(xs,ys)

        batch_train_loss.append(loss)
        batch_train_acc.append(acc)

    train_loss.append(np.mean(batch_train_loss))
    train_acc.append(np.mean(batch_train_acc))

    batch_val_loss = []
    batch_val_acc = []

    for i in range(val_idx[0], val_idx[1]):

        xs = x_samples[i].reshape(1,1,45)
        ys = y_samples[i].reshape(1, 45)

        loss, acc = model.test_on_batch(xs, ys)

        batch_val_loss.append(loss)
        batch_val_acc.append(acc)
    val_loss.append(np.mean(batch_val_loss))
    val_acc.append(np.mean(batch_val_acc))
    print('epoch {0:4d} train acc {1:0.3f} loss {2:0.3f} val acc {3:0.3f} loss {4:0.3f}'
          .format(epoch, np.mean(batch_train_acc), np.mean(batch_train_loss), np.mean(batch_val_acc), np.mean(batch_val_loss)))



#모델 history

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()


acc_ax = loss_ax.twinx()

loss_ax.plot(train_loss, 'y', label = 'train loss')
loss_ax.plot(val_loss, 'r', label = 'val loss')

acc_ax.plot(train_acc, 'b', label = 'train acc')
acc_ax.plot(val_acc, 'g', label = 'val acc')
#x 값
loss_ax.set_xlabel('epoch')
#y 값
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc = "upper left")
acc_ax.legend(loc = "lower left")

plt.show()

# 모델검증(시뮬레이션)
#상금을 평균 낸다
mean_prize = [  np.mean(rows[87:, 8]),
                np.mean(rows[87:, 9]),
                np.mean(rows[87:, 10]),
                np.mean(rows[87:, 11]),
                np.mean(rows[87:, 12])]
print(mean_prize)

#등수와 상금을 반환함
#순위에 오르지 못한경우에는 등수가 0으로 반환
def calc_reward(true_numbers, true_bonus, pred_numbers ):

    count = 0

    for ps in pred_numbers:
        if ps in true_numbers:
            count +=1

        if count ==6:
            return 0, mean_prize[0]
        elif count == 5 and true_bonus in pred_numbers:
            return 1, mean_prize[1]
        elif count == 5:
            return 2, mean_prize[2]
        elif count == 4:
            return 3, mean_prize[3]
        elif count == 3:
            return 4, mean_prize[4]
        return 5, 0
#여기 코드부터 해석
def gen_nubers_from_probability(nums_prob):

    ball_box = []

    for n in range(45):
        ball_count = int(nums_prob[n] * 100 +1)
        ball = np.full((ball_count), n+1)
        ball_box +=list(ball)

    selected_balls = []

    while True:

        if len(selected_balls) == 6:
            break
        ball_index =np.random.randint(len(ball_box), size = 1)[0]
        ball = ball_box[ball_index]

        if ball not in selected_balls:
            selected_balls.append(ball)
    return selected_balls








