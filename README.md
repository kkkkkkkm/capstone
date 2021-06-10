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


## 데이터


|피부질환   |개수|Train_data|Test_data|Validtion|
|---|:---:|:---:|:---:|:---:|
| 맨살 |1010|808|202|202|
| 티눈  |437|350|87|87|
| 사마귀  |1068|855|213|213|
| **total**  |**2515**|**2013**|**502**|**502**|

## Learning rate(acc)

Metric learning을 학습시킬 때와 Classifier를 학습 시킬 때 각각 learning rate를 다르게 주는 것이 가장 좋은 결과가 나왔다

![learing](https://user-images.githubusercontent.com/69561492/121500257-1af21d00-ca19-11eb-9795-a03710f3bfe2.PNG)

## 모델 결과 비교

### Naive 

다음 그림은 Metric Learning와 ImageNet 가중치를 적용하지 않고 데이터 증강만 시킨 모델의 학습 결과를 보여준다.  
티눈의 데이터 2배가량 부족한 이유 탓인지 티눈에 대한 학습이 이루어지지 않은 것을 확인할 수 있다.  
정확도는 60.8%가 나왔지만, F1 Score로 확인해본 결과 55.4%로 더욱 낮은 평가 결과가 나옴으로써 불균형한 학습이 이루어진 것을 확인할 수 있다.  

  
![naive_count](https://user-images.githubusercontent.com/69561492/121501446-390c4d00-ca1a-11eb-9bf7-8ae68caad3fb.PNG)


### InceptionV3

다음은 InceptionV3 Network에 ImageNet 가중치로 전이 학습, 데이터 증강, 미세조정을 시킨 결과이다.  
미세조정을 하기 전엔 64% 정확도로, Naive 모델을 사용했을 때와 큰 차이를 보이진 않았지만, 미세조정을 시킨 후 결과는 89.9%의 정확도를 보였고 F1 Score에서도 89.8%가 나옴으로써 이전 모델보단 균등한 학습이 이루어졌음을 확인할 수 있다. 그러나 우수한 결과라고 판단하긴 힘들다.

![inception_only_weight](https://user-images.githubusercontent.com/69561492/121503406-019ea000-ca1c-11eb-91ef-36e40e1ee18b.PNG)

아래는 Deep Metric Learning 기법의 하나인 Triplet Embedding(Semi-hard Loss 이용) 방식을 이용하여 학습을 시킨 결과이다. 정확도는 90.9%이고 F1 Score는 90.8%로 미세조정을 사용했을 때보다 미세하게 성능이 향상되었다.

![inception_triplet](https://user-images.githubusercontent.com/69561492/121503550-2266f580-ca1c-11eb-92f1-c27414bc7172.PNG)


### MobileNet
다음 표는 MobileNet에 적용한 학습 방법들과 그 결과를 정리해놓은 것이다.
결과를 보면 알 수 있듯 이전 네트워크를 사용한 것보다 MobileNet에서의 결과가 더 좋은 것을 볼 수 있다.   
 여러 방식을 시도해보다 데이터 증강을 시키는 것 보다 시키지 않았을 때 모델의 성능이 올라간다는 것을 발견했다.  
 Triplet Semi-hard Loss 이후의 방식들은 모두 데이터 증강을 적용하지 않았다.

|Method  |VAL|
|---|:---:|
|Classifier only |91.2%±1.3|
|Fine Tuning   |91.5%±1.5|
|  Triplet Semi-hard Loss |92.5%±1.5|
| Triplet Semi-hard Loss **(without augmentation)**|94.6%±1|
|  Triplet Hard Loss|93.0%±1|
|  Contrastive Loss |91.0%±2|
|  Lifted Structed Loss |95.8%±2|

  

다음은 Lifted Structed Loss를 이용하여 97%의 가장 높은 정확도를 기록한 결과이다.  

![Lifted STruct coun](https://user-images.githubusercontent.com/69561492/121505951-3f042d00-ca1e-11eb-9ca2-550374ec8206.PNG)


![lifted_norm](https://user-images.githubusercontent.com/69561492/121506042-56dbb100-ca1e-11eb-8f26-face91992e5d.png)

