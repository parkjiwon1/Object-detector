import os
os.add_dll_directory('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin')
import numpy as np
import tensorflow as tf
import cv2 as cv
import random
import matplotlib.pyplot as plt

# Generate training images and labels

N = 2000
H,W = 256,256 # Height, Width
train_img = np.zeros([N,H,W,3],dtype = np.uint8)
train_img.fill(255)
train_label_coord = np.zeros([N,2],dtype = np.int32) # N,4 -> N,2 / N,2
train_label_size = np.zeros([N,2],dtype = np.int32)


for n in range(N):
    x,y = random.randint(0,W-1),random.randint(0,H-1) # random location
    bh, bw = random.randint(int(W/16), int(W/4)),random.randint(int(H/16),int(H/4)) # random size
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

    train_label_coord[n,0] = x; train_label_coord[n,1] = y
    train_label_size[n,0] = bw; train_label_size[n,1] = bh

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
train_label_coord = train_label_coord.astype(np.float32)
train_label_coord[:,0] = train_label_coord[:,0]/W
train_label_coord[:,1] = train_label_coord[:,1]/H
train_label_size = train_label_size.astype(np.float32)
train_label_size[:,0] = train_label_size[:,0]/(W/4)
train_label_size[:,1] = train_label_size[:,1]/(H/4)

# Build Model
base_model = tf.keras.applications.VGG16(input_shape = [H,W,3], include_top = False,
                                             weights = 'imagenet')
x = base_model.output
x = tf.keras.layers.Flatten()(x)
out_coord = tf.keras.layers.Dense(2,activation = 'sigmoid',name='out_coord')(x)
out_size = tf.keras.layers.Dense(2,activation = 'sigmoid', name = 'out_size')(x)
model = tf.keras.Model(inputs = base_model.input,outputs = [out_coord,out_size]) # output = [(x,y),(bw,bh)]

model.summary()

# Train and save model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001),
              loss ={'out_coord':'mse','out_size':'mse'},
              loss_weights = {'out_coord':4,'out_size':1})

history = model.fit(x = train_img,y = [train_label_coord,train_label_size],
                    epochs = 10,batch_size = 50, validation_split = 0.25)

model.save('model')

# Show training history

"""
plt.figure(), plt.plot(history.history['loss'], 'b-', label='training')
plt.plot(history.history['val_loss'], 'r--', label='validation')
plt.title('Total loss'), plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend(), plt.show()
plt.figure(), plt.plot(history.history['out_coord_loss'], 'b-', label='training')
plt.plot(history.history['val_out_coord_loss'], 'r--', label='validation')
plt.title('Coordinate loss'), plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend(), plt.show()
plt.figure(), plt.plot(history.history['out_size_loss'], 'b-', label='training')
plt.plot(history.history['val_out_size_loss'], 'r--', label='validation')
plt.title('Size loss'), plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend(), plt.show()
"""

# Generate test images and labels
N = 6
H,W = 256,256
test_img = np.zeros([N,H,W,3],dtype = np.uint8)
test_img.fill(255)
test_label_coord = np.zeros([N,2],dtype = np.int32)
test_label_size = np.zeros([N,2],dtype = np.int32)


for n in range(N):
    x,y = random.randint(0,W-1),random.randint(0,H-1)
    bh, bw = random.randint(int(W / 16), int(W / 4)), random.randint(int(H / 16), int(H / 4))

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

    test_label_coord[n,0]=x; test_label_coord[n,1]= y
    test_label_size[n,0]=bw; test_label_size[n,1]= bh

    cv.rectangle(test_img[n], (x - int(bw / 2), y - int(bh / 2)), (x + int(bw / 2), y + int(bh / 2)),
                 color=(0, 255, 0), thickness=-1)

# Preprocess test images
test_img_ = tf.keras.applications.vgg16.preprocess_input(test_img)

# Predict object locations in test images
model = tf.keras.models.load_model('model')
pred_coord,pred_size = model.predict(test_img_)
pred_coord[:,0] = pred_coord[:,0]*W
pred_coord[:,1] = pred_coord[:,1]*H
pred_size[:,0] = pred_size[:,0]*(W/4)
pred_size[:,1] = pred_size[:,1]*(H/4)


# Display prediction results
for n in range(N):
    x = pred_coord[n,0].astype('int')
    y = pred_coord[n,1].astype('int')
    bw = pred_size[n,0].astype('int')
    bh = pred_size[n,1].astype('int')

    cv.circle(test_img[n],center = (x,y),radius = 2,color = (0,0,0), thickness = 2)
    cv.rectangle(test_img[n],(x-int(bw/2),y-int(bh/2)), (x+int(bw/2),y+int(bh/2)),
                 color = (0,0,0),thickness = 2)
    cv.imshow("test_img",test_img[n])
    cv.waitKey(0)
    cv.destroyWindow("test_img")
