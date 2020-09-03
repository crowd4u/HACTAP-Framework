# flake8: noqa
# coding: UTF-8
# 使う標準ライブラリ
import os
import pprint
import time
import urllib
import uuid

# よく使うライブラリ
from PIL import Image
import pandas as pd
import numpy as np

# 機械学習関連のライブラリ
import keras
# from keras import optimizers
# from keras.preprocessing.image import array_to_img, img_to_array, load_img
# from sklearn.model_selection import train_test_split
# from keras.models import model_from_json
# from keras.models import Sequential
# from keras.utils import np_utils
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers.core import Flatten, Dropout, Dense, Activation
# from keras.backend import tensorflow_backend as backend
import tensorflow
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.backend import clear_session
from tensorflow.python.keras.layers import Flatten, Dropout, Dense, Activation, Conv2D, MaxPooling2D
from tensorflow.python.keras import Sequential

height = 122
width = 110

class MindAIWorker:
    path = "Image/src/"

    def __init__(self):
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tensorflow.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        return

    def save_csv(self, url, status, path=path):
        image_data = self.__get_image(url)
        image_name = os.path.basename(os.path.splitext(url)[0])
        divided_image_data = self.__divide_image(
            image_data, status, image_name)

        if not os.path.isfile('train.csv'):
            with open('train.csv', mode='w') as f:
                f.write('image,status')

        df1 = pd.read_csv('train.csv')
        df2 = pd.DataFrame(divided_image_data,
                           columns=['image', 'status'])
        df = df1.append(df2)
        df = df.drop_duplicates()
        df.to_csv('train.csv', index=False)
        return 'csvに追加しました\n'

    def fit(self, x_train, y_train):
        # if not os.path.exists('train.csv'):
        #     return 'トレーニングファイルがありません。'

        # df = pd.read_csv('train.csv')
        # X = []
        # Y = df['status'].values

        # image_paths = df['image'].values
        # for image_path in image_paths:
        #     image = img_to_array(load_img(image_path, target_size=(122, 110)))
        #     X.append(image)

        # X = np.asarray(X)
        # Y = np.asarray(Y)
        # X = X.astype('float32')
        # X = X / 255.0

        # ステータス1,2,4をそれぞれ、0,1,2に変換

        # tmp1 = (Y == 1)
        # tmp2 = (Y == 2)
        # tmp4 = (Y == 4)
        # Y[tmp1] = 0
        # Y[tmp2] = 1
        # Y[tmp4] = 2

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_train = x_train.reshape(x_train.shape[0], height, width, 3)
        y_train = np_utils.to_categorical(y_train, 3)

        # try:
        # CNNモデル
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=(122, 110, 3), activation="relu")) # NOQA
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(3))
        model.add(Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"]) # NOQA
        model.fit(x_train, y_train, epochs=50)

        self.model = model

        # print('save the architecture of a model')
        # json_string = model.to_json()
        # open(os.path.join('./', 'cnn_model.json'), 'w').write(json_string)
        # print('save weights')
        # model.save_weights(os.path.join('./', 'cnn_model_weights.hdf5'))
        return
        # except:
        #     return'トレーニングに失敗しました。'
        # backend.clear_session()
        # return '学習に成功しました'

    def predict(self, x_test):

        # image_data = self.__get_image(url)
        # X_test = self.__divide_image_for_predict(image_data)
        # 学習結果を読み込む
        # model = model_from_json(open('cnn_model.json').read())
        # model.load_weights('cnn_model_weights.hdf5')
        # model.summary()
        # model.compile(loss="categorical_crossentropy", optimizer="SGD",
                    #   metrics=["accuracy"])

        # np.set_printoptions(threshold=np.inf)

        x_test = np.asarray(x_test)
        x_test = x_test.reshape(x_test.shape[0], height, width, 3)

        result = self.model.predict_classes(x_test)
        # result = np.reshape(result, (32, 32))

        # ステータス0,1,2をそれぞれ、1,2,4に変換
        # tmp0 = (result == 0)
        # tmp1 = (result == 1)
        # tmp2 = (result == 2)
        # result[tmp0] = 1
        # result[tmp1] = 2
        # result[tmp2] = 4

        # np.set_printoptions(threshold=np.inf)

        # result = np.array2string(result, separator=',')
        # backend.clear_session()
        return result

    def predict_proba(self, x_test):

        # image_data = self.__get_image(url)
        # X_test = self.__divide_image_for_predict(image_data)
        # 学習結果を読み込む
        # model = model_from_json(open('cnn_model.json').read())
        # model.load_weights('cnn_model_weights.hdf5')
        # model.summary()
        # model.compile(loss="categorical_crossentropy", optimizer="SGD",
                    #   metrics=["accuracy"])

        # np.set_printoptions(threshold=np.inf)

        x_test = np.asarray(x_test)
        x_test = x_test.reshape(x_test.shape[0], height, width, 3)

        result = self.model.predict_proba(x_test)
        # result = np.reshape(result, (32, 32))

        # ステータス0,1,2をそれぞれ、1,2,4に変換
        # tmp0 = (result == 0)
        # tmp1 = (result == 1)
        # tmp2 = (result == 2)
        # result[tmp0] = 1
        # result[tmp1] = 2
        # result[tmp2] = 4

        # np.set_printoptions(threshold=np.inf)

        # result = np.array2string(result, separator=',')
        # backend.clear_session()
        return result

    # private

    def __get_image(self, url):
        data = urllib.request.urlopen(url)
        return data

    def __divide_image(self, data, status, name):
        result = []
        # 最終的に、機械学習で読み込むときの分割数を指定
        last_div_h = 32
        last_div_w = 32
        im = Image.open(data)
        image_h = im.height
        image_w = im.width
        div_h = image_h / float(len(status))
        div_w = image_w / float(len(status[0]))
        x = 0
        y = 0
        divide_count1 = 1
        divide_count2 = 1
        for i in status:
            for label in i:
                divided_image = im.crop((x, y, x + div_w, y + div_h))
                divided_image_h = divided_image.height
                divided_image_w = divided_image.width
                div_h_ = divided_image_h / float(last_div_h/len(status))
                div_w_ = divided_image_w / float(last_div_w/len(status[0]))
                x_ = 0
                y_ = 0
                for _ in range(int(last_div_h / len(status))):
                    for _ in range(int(last_div_w / len(status[0]))):
                        filename = self.path + name + '_' + \
                            str(divide_count1) + '_' + \
                            str(divide_count2) + '_' + str(label) + '.png'
                        divided_divided_image = divided_image.crop(
                            (x_, y_, x_ + div_w_, y_ + div_h_))
                        if label != 3 and label != -1:
                            result.append([filename, label])
                            if not os.path.exists(filename):
                                divided_divided_image.save(filename, 'PNG')
                        x_ += div_w_
                        divide_count2 += 1
                    x_ = 0
                    y_ += div_h_
                x += div_w
                divide_count1 += 1
                divide_count2 = 1
            x = 0
            y += div_h
        return result

    def __divide_image_for_predict(self, data):
        X = []
        image_size = 50
        # 最終的に、機械学習で読み込むときの分割数を指定
        last_div_h = 32
        last_div_w = 32
        im = Image.open(data)
        image_h = im.height
        image_w = im.width
        div_h = image_h / last_div_h
        div_w = image_w / last_div_w
        x = 0
        y = 0
        for _ in range(last_div_h):
            for _ in range(last_div_w):
                divided_image = im.crop((x, y, x + div_w, y + div_h))
                divided_image = divided_image.convert("RGB")
                divided_image = divided_image.resize((110, 122))
                data = np.asarray(divided_image)
                X.append(data)
                x += div_w
            x = 0
            y += div_h
        X = np.asarray(X)
        X = X.astype('float32')
        X = X / 255.0
        return X