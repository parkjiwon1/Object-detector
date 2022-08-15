import os
os.add_dll_directory('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin')
import numpy as np
import tensorflow as tf
import cv2 as cv
import random
import matplotlib.pyplot as plt

# Generate training images and labels

N = 4000
H,W = 256,256 # Height, Width
train_img = np.zeros([N,H,W,3],dtype = np.uint8)
train_img.fill(255)
train_label = np.zeros([N,2],dtype = np.int32)

bh, bw = 50,50
for n in range(N):
    x,y = random.randint(0,W-1),random.randint(0,H-1)
    if (x-bw/2 <0):
        x = x-(x-bw/2)
    elif (x+bw/2>W-1):
        x = x-(x+bw/2-(W-1))
    if (y-bh/2 <0):
        y = y-(y-bh/2)
    elif (y+bh/2 > H-1):
        y = y-(y+bh/2-(H-1))
    x = int(x)
    y = int(y)
    train_label[n,0] = x
    train_label[n,1] = y
    cv.rectangle(train_img[n],(x-int(bw/2),y-int(bh/2)), (x+int(bw/2),y+int(bh/2)),
                 color = (0,255,0),thickness = -1)
#Display some images
"""
for n in range(3):
    cv.imshow("train_img",train_img[n])
    cv.waitKey(0)
    cv.destroyWindow("train_img")
"""
# Preprocess data
train_img = tf.keras.applications.vgg16.preprocess_input(train_img)
train_label = train_label.astype(np.float32)
train_label[:,0] = train_label[:,0]/W
train_label[:,1] = train_label[:,1]/H

# Build Model
base_model = tf.keras.applications.VGG16(input_shape = [H,W,3], include_top = False,
                                             weights = 'imagenet')
x = base_model.output
x = tf.keras.layers.Flatten()(x)
predictions = tf.keras.layers.Dense(2,activation= 'sigmoid')(x)
model = tf.keras.Model(inputs = base_model.input,outputs = predictions)

model.summary()

# Train and save model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001), loss = 'mse')
history = model.fit(train_img,train_label,epochs = 1,batch_size = 10, validation_split = 0.25)
model.save('model')

# Show training history
"""
plt.plot(history.history['loss'],'b-',label = 'training')
plt.plot(history.history['val_loss'],'r-',label = 'validation')
plt.xlabel('Epoch'), plt.ylabel('Loss'),plt.legend()
plt.show()
"""
# Generate test images and labels
N = 5
H,W = 256,256
test_img = np.zeros([N,H,W,3],dtype = np.uint8)
test_img.fill(255)
test_label = np.zeros([N,2],dtype = np.int32)

bh,bw = 50,50
for n in range(N):
    x,y = random.randint(0,W-1),random.randint(0,H-1)
    if (x - bw / 2 < 0):
        x = x - (x - bw / 2)
    elif (x + bw / 2 > W - 1):
        x = x - (x + bw / 2 - (W - 1))
    if (y - bh / 2 < 0):
        y = y - (y - bh / 2)
    elif (y + bh / 2 > H - 1):
        y = y - (y + bh / 2 - (H - 1))
    x = int(x)
    y = int(y)
    train_label[n, 0] = x
    train_label[n, 1] = y
    cv.rectangle(test_img[n], (x - int(bw / 2), y - int(bh / 2)), (x + int(bw / 2), y + int(bh / 2)),
                 color=(0, 255, 0), thickness=-1)

# Preprocess test images
test_img_ = tf.keras.applications.vgg16.preprocess_input(test_img)

# Predict object locations in test images
model = tf.keras.models.load_model('model')
out = model.predict(test_img_)
out[:,0] = out[:,0]*W
out[:,1] = out[:,1]*H

# Display prediction results
for n in range(N):
    x = out[n,0].astype('int')
    y = out[n,1].astype('int')
    cv.circle(test_img[n],center = (x,y),radius = 2,color = (0,0,0), thickness = 2)
    cv.imshow("test_img",test_img[n])
    cv.waitKey(0)
    cv.destroyWindow("test_img")
