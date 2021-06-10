# 7조 캡스톤 주제: 이미지 처리를 통한 사마귀와 티눈 분류 
## 코드 설명 

관련 모듈 import
checkpoint는 callback 함수를 이용해 모델을 저장 할 위치

```python
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import random
import seaborn as sns
import sklearn.metrics
import pandas as pd
from sklearn.decomposition import PCA
checkpoint_path2 = '/home/gangmin_data/capstone_data/checkpoint2'
checkpoint_path = '/home/gangmin_data/capstone_data/checkpoint'
import tensorflow_addons as tfa
import zipfile
```

tensorflow_addons 이 없다면 설치
```pythone
pip install tensorflow-addons
```

glob를 이용해 모든 데이터의 경로 가져오기, jpg로 불러왔을 때 JPG 확장자인 경로는 가져오지 못해서 따로 선언하였음 
```pythone
data_list = glob('/home/gangmin_data/capstone_data2/training/*/*/*.jpg')
data_list2 = glob('/home/gangmin_data/capstone_data2/training/*/*/*.JPG')
test_data = glob('/home/gangmin_data/capstone_data2/test/*/*/*.jpg')
test_data2 = glob('/home/gangmin_data/capstone_data2/test/*/*/*.JPG')
test_data = test_data + test_data2
data_list = test_data +data_list
```

path에서 label 분리  
(/home/gangmin_data/capstone_data2/training/1/1_train/20210408_192946.jpg 경로에서 label 1만 떼어내기 위한 함수)
```python
def get_label_from_path(path): #label 분리 
    list_label = []
    for i in range(len(path)):
        list_label.append(int(path[i].split('/')[-3]))
    return list_label
```

데이터 전처리 함수 float32로 cast 후, 데이터 표현의 폭을 넓히기 위해 0 ~ 1 이 아닌 -1 ~ 1 로 정규화 해주었음.  
몇몇 데이터의 차원이 3, 4 차원이 아닌 2차원 데이터들이 존재하여 resize 단계에 오류가 생기는 경우가 있었는데, 그런 데이터는 우리가 바라는 format이 아니기에 삭제  
```python
IMG_SIZE = 128 # 모든 이미지는 128x128으로 크기가 조정됩니다

def format_example(path):   #전처리
    image = np.array(Image.open(path))
    image = tf.cast(image, tf.float32)
    image = (image/127.5) -1
    if image.ndim != (3 or 4):
        print(path, 'ndim =', image.ndim)
        os.remove(path)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image
```

NHWC형태로 데이터 생성
```python
batch_size = 32
data_height = 128
data_width = 128
channel_n = 3

def make_Batch_image(data_list):
    batch_image = np.zeros((len(data_list), data_height, data_width,channel_n))
    for n in range(len(data_list)):
        path = data_list[n]
        image = format_example(path)
        if image.shape != (data_height,data_width, 3):
            os.remove(path)
        else:
            batch_image[n,:,:,:] = image
    return batch_image
```

데이터셋으로 생성
```python
def make_dataset(image, label):
    image = tf.cast(image, dtype = 'float32')
    label = tf.cast(label, dtype = 'uint8')
    ds = tf.data.Dataset.from_tensor_slices( (image, label)).shuffle(1971).batch(32)
    return ds
```

Augmentation
```python
def augment(image, label):
   # Random crop back to the original size
    a = random.uniform(0.1,0.5)
    b = random.randrange(0, 359)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.adjust_saturation(image, 3)
    image = tf.image.adjust_brightness(image, a)
    image = tfa.image.rotate(image, angles = b, fill_mode = 'reflect') # Random brightnes
    
    return image, label

train_ds = (train_ds.shuffle(1975).map(augment, 
                                       num_parallel_calls = tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE))
```


 Ensemble 기법 사용, 처음 Triplet Loss로 학습 할 때 loss가 0.4 ~ 0.8 까지 값의 변동이 컸음.
 loss가 0.4에 수렴하는 경우엔 이후 분류 학습에서 괜찮은 결과가 나왔지만 0.8에서 끝나는 경우는 결과가 잘 나오지 않았음.  
 loss의 변동값을 줄이기 위하여 사용
```python

modelcheck = tf.keras.callbacks.ModelCheckpoint(filepath= checkpoint_path ,monitor= 'val_loss', s
                                                ave_best_only= True, mode = 'min', save_freq= 'epoch' )
callback = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= 3)

def get_ensemble():
    inputs = tf.keras.Input(shape = IMG_SHAPE)
    x = transfer1_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024,)(x)
    outputs = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return tf.keras.Model(inputs, outputs)

model1 = get_ensemble()
model2 = get_ensemble()
model3 = get_ensemble()

inputs = tf.keras.Input(shape = IMG_SHAPE)

y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)

outputs = tf.keras.layers.average([y1,y2,y3])

ensemble_model = tf.keras.Model(inputs = inputs,outputs = outputs)

ensemble_model.compile(loss = tfa.losses.LiftedStructLoss(),
             optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0005),)

ensemble_model.fit(
    train_ds,
    validation_data = test_ds,
    callbacks= [callback, modelcheck],
    epochs=100)

```

Confusion Matrix 생성
```python
LABELS = ['Warts', 'Corns', 'Noraml Skin']

def confusion_mat(model,test_image,test_label):
    predictions = model.predict(test_image)
    arg = np.argmax(predictions, axis = 1)
    tf.math.confusion_matrix(labels = test_label, predictions = arg)
    sns.heatmap(tf.math.confusion_matrix(labels = test_label, predictions = arg), 
                xticklabels= LABELS, yticklabels= LABELS, cmap = 'Blues', annot = True, fmt = "d" )

    metric = tfa.metrics.F1Score(num_classes = 3, threshold = None, average = 'weighted' )
    category = tf.keras.utils.to_categorical(test_label, num_classes = 3)
    metric.update_state(category, predictions)

    result = metric.result()
    print(result)

```

Nomalize 시킨 Confusion Matrix
```python
def confusion_norm(model,test_image,test_label ):
    predictions = model.predict(test_image)
    arg = np.argmax(predictions, axis = 1)
    normalize= sklearn.metrics.confusion_matrix(test_label, arg, normalize='true')
    sns.heatmap(normalize, cmap = 'Blues', annot = True, fmt = "f")
```


## 결과 
