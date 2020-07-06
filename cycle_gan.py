import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
print(tf.__version__)
print(np.__version__)
makeup_path="makeup_datset/data/makeup_with_labels/yes_makeup"
no_makeup_path="makeup_datset/data/makeup_with_labels/no_makeup"
# loading the data and all
# makeup_data
makeup_data=[]
for i in os.listdir(makeup_path):
    makeup_data.append(cv2.imread(os.path.join(makeup_path,i),cv2.COLOR_BGR2YUV))

# no makeup
nomakeup_data=[]
for i in os.listdir(no_makeup_path):
    nomakeup_data.append(cv2.imread(no_makeup_path+"/"+i,cv2.COLOR_BGR2YUV))

makeup_data=np.array(makeup_data)
nomakeup_data=np.array(nomakeup_data)

makeup_data[0].shape
nomakeup_data[0].shape
plt.imshow(nomakeup_data[0])
plt.show()

# up means with makeup
# down means without makeup
gen_up_down=tf.keras.Sequential(
    [   
        tf.keras.layers.Conv2D(32,(3,3),padding="same",input_shape=(64,64,1),activation="relu"),
        tf.keras.layers.MaxPooling2D(strides=(2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(12,(3,3),padding="same",activation="relu"),
        tf.keras.layers.MaxPooling2D(strides=(2,2)),
        tf.keras.layers.Conv2D(12,(3,3),padding="same",activation="relu"),
        tf.keras.layers.Conv2DTranspose(12,(3,3),padding='same',activation="relu",strides=(2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(12,(3,3),padding='same',activation="relu",strides=(2,2)),
        tf.keras.layers.Conv2DTranspose(1,(3,3),padding="same",activation="relu")
        

        
    ]
)
ans=gen_up_down.predict(makeup_data[0].reshape(1,64,64,1))
plt.imshow(ans.reshape(64,64))
plt.show()

gen_down_up=tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32,(3,3),padding="same",input_shape=(64,64,1),activation="relu"),
        tf.keras.layers.MaxPooling2D(strides=(2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(12,(3,3),padding="same",activation="relu"),
        tf.keras.layers.MaxPooling2D(strides=(2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(12,(3,3),padding="same",activation="relu"),
        tf.keras.layers.Conv2DTranspose(12,(3,3),padding='same',activation="relu",strides=(2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(12,(3,3),padding='same',activation="relu",strides=(2,2)),
        tf.keras.layers.Conv2DTranspose(1,(3,3),padding="same",activation="relu")

        
    ]
)
print(gen_down_up.output_shape)
desc_a=tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(64,64,1)),
        tf.keras.layers.Conv2D(32,(3,3),padding='same',activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(12,(3,3),padding="same",activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32,activation="relu"),
        tf.keras.layers.Dense(10,activation='relu'),
        tf.keras.layers.Dense(1,activation="relu")
 

    ]
)

# losses 
cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)
def desc_loss(real,fake):
    real_loss=cross_entropy(tf.ones_like(real),real)
    fake_loss=cross_entropy(tf.ones_like(fake),fake)
    return real_loss+fake_loss

def gen_loss(fake):
    return cross_entropy(tf.ones_like(fake),fake)

def cycle_loss(X,x):
    return tf.reduce_mean(tf.abs(X-X))

# optimizers 
gen_up_down_opti=tf.keras.optimizers.Adam()
gen_down_up_opti=tf.keras.optimizers.Adam()
desc_opti=tf.keras.optimizers.Adam()

# first seeing that if the gan could work
def training_module(with_make,without_make):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as desc_tape:
        fake=gen_up_down(with_make,training=True)
        fake_prediction=desc_a(fake,training=True)
        real_predictions=desc_a(without_make,training=True)
        loss=desc_loss(real_predictions,fake_prediction)



