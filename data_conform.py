#!/usr/bin/env python
# coding: utf-8

# In[5]:


#########################################


# In[1]:


from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split


# In[2]:


import tensorflow_addons as tfa


# In[3]:


data_list = glob('/home/gangmin_data/capstone_data/training/*/*.jpg')
data_list2 = glob('/home/gangmin_data/capstone_data/training/1/*.JPG')


# In[4]:


data_list = data_list + data_list2


# In[5]:


test_data = glob('/home/gangmin_data/capstone_data/test/*/*.jpg')


# In[6]:


print(len(data_list))
print(len(test_data))


# In[7]:


train, valid = train_test_split(data_list, test_size = 0.2,)


# In[8]:


train_path = train
valid_path = valid
test_path = test_data


# In[9]:


def get_label_from_path(path):
    list_label = []
    for i in range(len(path)):
        list_label.append(int(path[i].split('/')[-2]))
    return list_label


# In[10]:


IMG_SIZE = 160 # 모든 이미지는 160x160으로 크기가 조정됩니다

def format_example(path):
    image = np.array(Image.open(path))
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    if image.ndim != (3 or 4):
        print(path, 'ndim =', image.ndim)
        os.remove(path)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image


# In[11]:


batch_size = 32

batch_per_epoch =  len(data_list) // batch_size 

label = get_label_from_path(train_path)

val_label = get_label_from_path(valid_path)

test_label = get_label_from_path(test_path)


# In[12]:


print(label)
print(len(label))


# In[13]:


print(test_label)
print(len(test_label))
print(test_label.count(1))


# In[14]:


batch_size = 32
data_height = 160
data_width = 160
channel_n = 3

def make_Batch_image(data_list):
    batch_image = np.zeros((len(data_list), data_height, data_width,channel_n))
    for n in range(len(data_list)):
        path = data_list[n]
        image = format_example(path)
        if image.shape != (160,160,3):
            os.remove(path)
        else:
            batch_image[n,:,:,:] = image
    return batch_image


# In[15]:


train_image = make_Batch_image(train_path)
test_image = make_Batch_image(test_data)
valid_image = make_Batch_image(valid_path)


# In[16]:


print(train_image.shape)


# In[17]:


def make_dataset(image, label):
    image = tf.cast(image, dtype = 'float32')
    label = tf.cast(label, dtype = 'uint8')
    ds = tf.data.Dataset.from_tensor_slices( (image, label)).shuffle(20).batch(32)
    return ds


# In[18]:


train_ds = make_dataset(train_image, label)
test_ds = make_dataset(test_image, test_label)
valid_ds = make_dataset(valid_image, val_label)


# In[19]:


train_ds
for image_batch, label_batch in train_ds.take(1):
    print(image_batch.shape)
    print(label_batch.shape)


# In[20]:


IMG_SIZE = 160

def augment(image, label):
   # Random crop back to the original size
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.adjust_saturation(image,3)
    image = tf.image.random_brightness(image, max_delta = 0.4) # Random brightness
    return image, label


# In[21]:


train_ds = (train_ds.shuffle(32).map(augment, num_parallel_calls = tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE))


# In[22]:


for img, label in train_ds.take(1):
    print(img)
    print(label)


# In[23]:


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
inception_model = tf.keras.applications.MobileNet(include_top = False, weights = 'imagenet', input_shape =(IMG_SHAPE))
fine_inception = tf.keras.applications.MobileNet(include_top = False, weights = 'imagenet', input_shape =(IMG_SHAPE))


# In[24]:


inception_model.summary()


# In[25]:


triplet_model = tf.keras.Sequential([
    inception_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)), # L2 normalize embeddings
])


# In[26]:


triplet_model.compile(loss = tfa.losses.TripletSemiHardLoss(),
             optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001),)


# In[27]:


checkpoint_path = '/home/gangmin_data/capstone_data/checkpoint'
modelcheck = tf.keras.callbacks.ModelCheckpoint(filepath= checkpoint_path ,monitor= 'val_loss', save_best_only= True, mode = 'min', save_freq= 'epoch' )
callback = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= 3)
triplet_model.fit(
    train_ds,
    validation_data = valid_ds,
    callbacks= [callback,modelcheck],
    epochs=100)

triplet_model = tf.keras.models.load_model(checkpoint_path)
triplet_model.evaluate(valid_ds, batch_size=128, verbose=1)


# In[28]:


triplet_model = tf.keras.models.load_model(checkpoint_path)
triplet_model.evaluate(valid_ds, batch_size=128, verbose=1)


# In[29]:


triplet_model.trainble = False


# In[30]:


get_ipython().system('mkdir -p saved_model')


triplet_model_result = tf.keras.Sequential([
    triplet_model,
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(3)
]) 
checkpoint_path2 = '/home/gangmin_data/capstone_data/checkpoint2'
modelcheck2 = tf.keras.callbacks.ModelCheckpoint(filepath= checkpoint_path2 ,monitor= 'val_loss', save_best_only= True, mode = 'min', save_freq= 'epoch' )

triplet_model_result.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True),
             optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001),
             metrics = ['accuracy'])

triplet_model_result.fit(train_ds, validation_data = valid_ds, epochs = 100, callbacks = [callback, modelcheck2])


triplet_model_result.evaluate(test_ds)

triplet_model_result.save('/home/gangmin_data/capstone_data/my_model')


# In[31]:


triplet_model_result =tf.keras.models.load_model(checkpoint_path2)
triplet_model_result.evaluate(test_ds)


# In[46]:


load_model =  tf.keras.models.load_model('/home/gangmin_data/capstone_data/my_model')
load_model.evaluate(test_ds)


# In[32]:


############## classifier 만 조정


# In[33]:


fine_inception.trainable = False


# In[35]:


model = tf.keras.Sequential([
    fine_inception,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.Dense(3),
])


# In[36]:


model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True),
             optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001),
             metrics = ['accuracy'])


# In[37]:


model.summary()


# In[38]:


callback = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= 3)
model.fit(train_ds, validation_data= valid_ds, epochs = 100, callbacks= [callback])


# In[39]:


model.evaluate(test_ds)


# In[40]:


fine_inception.trainable = True


# In[41]:


print('number of layers :', len(fine_inception.layers))


# In[42]:


fine_layer = 60


# In[43]:


for layer in fine_inception.layers[:fine_layer]:
    layer.trainable = False


# In[44]:


model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True),
             optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001),
             metrics = ['accuracy'])


# In[45]:


model.fit(train_ds, validation_data= valid_ds, epochs = 100, callbacks= [callback])


# In[68]:


model.evaluate(test_ds)


# In[ ]:





# In[ ]:





# In[ ]:




