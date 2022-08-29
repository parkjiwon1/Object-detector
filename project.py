import os
os.add_dll_directory('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin')
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

def conf_loss_func(y_true,y_pred):
    obj_mask=tf.cast(y_true[:,:,:,0],tf.bool)
    noobj_mask=tf.logical_not(obj_mask)
    sqrerr = tf.square(y_true-y_pred)
    loss_obj_conf=tf.reduce_mean(tf.boolean_mask(sqrerr,obj_mask))
    loss_noobj_conf=tf.reduce_mean(tf.boolean_mask(sqrerr,noobj_mask))
    loss_conf = loss_obj_conf + loss_noobj_conf
    return loss_conf

def coord_loss_func(y_true,y_pred):
    obj_mask=tf.cast(y_true[:,:,:,0],tf.bool)
    y_true=y_true[:,:,:,1:3]
    loss_coord=tf.reduce_mean(tf.boolean_mask(tf.square(y_true-y_pred),obj_mask))
    return loss_coord

def size_loss_func(y_true,y_pred):
    obj_mask=tf.cast(y_true[:,:,:,0],tf.bool)
    y_true=y_true[:,:,:,1:3]
    loss_size=tf.reduce_mean(tf.boolean_mask(tf.square(y_true-y_pred),obj_mask))
    return loss_size

def class_loss_func(y_true,y_pred):
    obj_mask=tf.cast(y_true[:,:,:,0],tf.bool)
    y_true=y_true[:,:,:,1]
    loss_class=tf.reduce_mean(tf.boolean_mask(
                    tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred),
                    obj_mask))
    return loss_class


N= 100 # 100개의 train datasets
H,W=416,416
GHN,GWN=11,11
GH,GW=int(H/GHN),int(W/GWN)
train_img=np.zeros([N,H,W,3],dtype=np.uint8)
train_label_conf=np.zeros([N,GHN,GWN,1],dtype=np.float32)
train_label_coord=np.zeros([N,GHN,GWN,2],dtype=np.float32)
train_label_size=np.zeros([N,GHN,GWN,2],dtype=np.float32)
train_label_class=np.zeros([N,GHN,GWN,1],dtype=np.float32)

df = pd.read_csv('BCCD/train/_annotations.csv',header = None)

train_img[0] = cv.imread('BCCD/train/0.jpg')
train_img[1] = cv.imread('BCCD/train/1.jpg')
train_img[2] = cv.imread('BCCD/train/2.jpg')
train_img[3] = cv.imread('BCCD/train/3.jpg')
train_img[4] = cv.imread('BCCD/train/4.jpg')
train_img[5] = cv.imread('BCCD/train/5.jpg')
train_img[6] = cv.imread('BCCD/train/6.jpg')
train_img[7] = cv.imread('BCCD/train/7.jpg')
train_img[8] = cv.imread('BCCD/train/8.jpg')
train_img[9] = cv.imread('BCCD/train/9.jpg')
train_img[10] = cv.imread('BCCD/train/10.jpg')
train_img[11] = cv.imread('BCCD/train/11.jpg')
train_img[12] = cv.imread('BCCD/train/12.jpg')
train_img[13] = cv.imread('BCCD/train/13.jpg')
train_img[14] = cv.imread('BCCD/train/14.jpg')
train_img[15] = cv.imread('BCCD/train/15.jpg')
train_img[16] = cv.imread('BCCD/train/16.jpg')
train_img[17] = cv.imread('BCCD/train/17.jpg')
train_img[18] = cv.imread('BCCD/train/18.jpg')
train_img[19] = cv.imread('BCCD/train/19.jpg')
train_img[20] = cv.imread('BCCD/train/20.jpg')
train_img[21] = cv.imread('BCCD/train/21.jpg')
train_img[22] = cv.imread('BCCD/train/22.jpg')
train_img[23] = cv.imread('BCCD/train/23.jpg')
train_img[24] = cv.imread('BCCD/train/24.jpg')
train_img[25] = cv.imread('BCCD/train/25.jpg')
train_img[26] = cv.imread('BCCD/train/26.jpg')
train_img[27] = cv.imread('BCCD/train/27.jpg')
train_img[28] = cv.imread('BCCD/train/28.jpg')
train_img[29] = cv.imread('BCCD/train/29.jpg')
train_img[30] = cv.imread('BCCD/train/30.jpg')
train_img[31] = cv.imread('BCCD/train/31.jpg')
train_img[32] = cv.imread('BCCD/train/32.jpg')
train_img[33] = cv.imread('BCCD/train/33.jpg')
train_img[34] = cv.imread('BCCD/train/34.jpg')
train_img[35] = cv.imread('BCCD/train/35.jpg')
train_img[36] = cv.imread('BCCD/train/36.jpg')
train_img[37] = cv.imread('BCCD/train/37.jpg')
train_img[38] = cv.imread('BCCD/train/38.jpg')
train_img[39] = cv.imread('BCCD/train/39.jpg')
train_img[40] = cv.imread('BCCD/train/40.jpg')
train_img[41] = cv.imread('BCCD/train/41.jpg')
train_img[42] = cv.imread('BCCD/train/42.jpg')
train_img[43] = cv.imread('BCCD/train/43.jpg')
train_img[44] = cv.imread('BCCD/train/44.jpg')
train_img[45] = cv.imread('BCCD/train/45.jpg')
train_img[46] = cv.imread('BCCD/train/46.jpg')
train_img[47] = cv.imread('BCCD/train/47.jpg')
train_img[48] = cv.imread('BCCD/train/48.jpg')
train_img[49] = cv.imread('BCCD/train/49.jpg')
train_img[50] = cv.imread('BCCD/train/50.jpg')
train_img[51] = cv.imread('BCCD/train/51.jpg')
train_img[52] = cv.imread('BCCD/train/52.jpg')
train_img[53] = cv.imread('BCCD/train/53.jpg')
train_img[54] = cv.imread('BCCD/train/54.jpg')
train_img[55] = cv.imread('BCCD/train/55.jpg')
train_img[56] = cv.imread('BCCD/train/56.jpg')
train_img[57] = cv.imread('BCCD/train/57.jpg')
train_img[58] = cv.imread('BCCD/train/58.jpg')
train_img[59] = cv.imread('BCCD/train/59.jpg')
train_img[60] = cv.imread('BCCD/train/60.jpg')
train_img[61] = cv.imread('BCCD/train/61.jpg')
train_img[62] = cv.imread('BCCD/train/62.jpg')
train_img[63] = cv.imread('BCCD/train/63.jpg')
train_img[64] = cv.imread('BCCD/train/64.jpg')
train_img[65] = cv.imread('BCCD/train/65.jpg')
train_img[66] = cv.imread('BCCD/train/66.jpg')
train_img[67] = cv.imread('BCCD/train/67.jpg')
train_img[68] = cv.imread('BCCD/train/68.jpg')
train_img[69] = cv.imread('BCCD/train/69.jpg')
train_img[70] = cv.imread('BCCD/train/70.jpg')
train_img[71] = cv.imread('BCCD/train/71.jpg')
train_img[72] = cv.imread('BCCD/train/72.jpg')
train_img[73] = cv.imread('BCCD/train/73.jpg')
train_img[74] = cv.imread('BCCD/train/74.jpg')
train_img[75] = cv.imread('BCCD/train/75.jpg')
train_img[76] = cv.imread('BCCD/train/76.jpg')
train_img[77] = cv.imread('BCCD/train/77.jpg')
train_img[78] = cv.imread('BCCD/train/78.jpg')
train_img[79] = cv.imread('BCCD/train/79.jpg')
train_img[80] = cv.imread('BCCD/train/80.jpg')
train_img[81] = cv.imread('BCCD/train/81.jpg')
train_img[82] = cv.imread('BCCD/train/82.jpg')
train_img[83] = cv.imread('BCCD/train/83.jpg')
train_img[84] = cv.imread('BCCD/train/84.jpg')
train_img[85] = cv.imread('BCCD/train/85.jpg')
train_img[86] = cv.imread('BCCD/train/86.jpg')
train_img[87] = cv.imread('BCCD/train/87.jpg')
train_img[88] = cv.imread('BCCD/train/88.jpg')
train_img[89] = cv.imread('BCCD/train/89.jpg')
train_img[90] = cv.imread('BCCD/train/90.jpg')
train_img[91] = cv.imread('BCCD/train/91.jpg')
train_img[92] = cv.imread('BCCD/train/92.jpg')
train_img[93] = cv.imread('BCCD/train/93.jpg')
train_img[94] = cv.imread('BCCD/train/94.jpg')
train_img[95] = cv.imread('BCCD/train/95.jpg')
train_img[96] = cv.imread('BCCD/train/96.jpg')
train_img[97] = cv.imread('BCCD/train/97.jpg')
train_img[98] = cv.imread('BCCD/train/98.jpg')
train_img[99] = cv.imread('BCCD/train/99.jpg')

obj_cnt = np.zeros([N],dtype = np.uint8)

for i in range(N): # obj_cnt
    obj_cnt[i] = df[8][i+1]

offset = 0
for n in range(N):
    M = obj_cnt[n]

    for m in range(M):
        if (df[3][m+1+offset] == 'RBC'): obj_class = 0
        elif (df[3][m+1+offset] == 'WBC'): obj_class = 1
        elif (df[3][m+1+offset] == 'Platelets'): obj_class = 2 # 0: RBC, 1: WBC, 2: Platelets

        x,y= (int(df[6][m+1+offset])+int(df[4][m+1+offset]))/2,(int(df[7][m+1+offset])+int(df[5][m+1+offset]))/2

        bw,bh= int(df[6][m+1+offset])-int(df[4][m+1+offset]), int(df[7][m+1+offset])-int(df[5][m+1+offset])

        x=int(x); y=int(y)


        i,j=int(x/GW),int(y/GH)
        if ((i >= 11) | (j >= 11)):
            i -= 1
            j -= 1

        train_label_conf[n,j,i] = 1
        train_label_coord[n,j,i,0:2]=((x-i*GW)/GW),((y-j*GH)/GH) # Normalized
        train_label_size[n,j,i,0:2]=(bw/(W/4)),(bh/(H/4)) # Normalized
        train_label_class[n,j,i,0]= obj_class
    offset += M


#training
train_img = tf.keras.applications.inception_v3.preprocess_input(train_img)


base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=[H,W,3], include_top=False,
                                        weights='imagenet')
x = base_model.output

out_conf=tf.keras.layers.Conv2D(1,(1,1),padding='same',activation='sigmoid',
                                name = 'out_conf')(x)

out_coord=tf.keras.layers.Conv2D(2,(1,1),padding='same',activation='sigmoid',
                                name='out_coord')(x)
out_size=tf.keras.layers.Conv2D(2,(1,1),padding='same',activation='sigmoid',
                                name='out_size')(x)
out_class=tf.keras.layers.Conv2D(3,(1,1),padding='same',activation='softmax',
                                name='out_class')(x)

model=tf.keras.Model(inputs=base_model.input,
                    outputs=[out_conf,out_coord,out_size,out_class])

model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), #매우 적은 수의 train dataset이라 learning rate를 조금 더 낮춤
        loss={'out_conf':conf_loss_func,'out_coord':coord_loss_func,
              'out_size':size_loss_func,'out_class':class_loss_func},
        loss_weights={'out_conf':3,'out_coord':1,'out_size':2,'out_class':4}) #class에 가중치를 조금 더 줌

history = model.fit(x=train_img,
                    y={'out_conf':train_label_conf,
                    'out_coord':tf.concat([train_label_conf, train_label_coord],-1),
                    'out_size':tf.concat([train_label_conf, train_label_size],-1),
                    'out_class':tf.concat([train_label_conf, train_label_class],-1)},
                    epochs=200,batch_size=5,validation_split=0.1)

model.save('model')


# Show training history
plt.figure()
plt.plot(history.history['loss'], 'b-'
, label='training')
plt.plot(history.history['val_loss'], 'r--'
, label='validation')
plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend()
plt.show()


# test images and labels
N=10 # 10 개의 test datasets
H,W=416,416
GHN,GWN=11,11
GH,GW=int(H/GHN),int(W/GWN)
test_img=np.zeros([N,H,W,3],dtype=np.uint8)


test_img[0] = cv.imread('BCCD/test/0.jpg')
test_img[1] = cv.imread('BCCD/test/1.jpg')
test_img[2] = cv.imread('BCCD/test/2.jpg')
test_img[3] = cv.imread('BCCD/test/3.jpg')
test_img[4] = cv.imread('BCCD/test/4.jpg')
test_img[5] = cv.imread('BCCD/test/5.jpg')
test_img[6] = cv.imread('BCCD/test/6.jpg')
test_img[7] = cv.imread('BCCD/test/7.jpg')
test_img[8] = cv.imread('BCCD/test/8.jpg')
test_img[9] = cv.imread('BCCD/test/9.jpg')

test_label_conf=np.zeros([N,GHN,GWN,1],dtype=np.float32)
test_label_coord=np.zeros([N,GHN,GWN,2],dtype=np.float32)
test_label_size=np.zeros([N,GHN,GWN,2],dtype=np.float32)
test_label_class=np.zeros([N,GHN,GWN,3],dtype=np.float32)

df2 = pd.read_csv('BCCD/test/_annotations.csv',header = None)

for i in range(N): # obj_cnt
    obj_cnt[i] = df2[8][i+1]

offset = 0
for n in range(N):
    M = obj_cnt[n]

    for m in range(M):
        if (df2[3][m+1+offset] == 'RBC'): obj_class = 0
        elif (df2[3][m+1+offset] == 'WBC'): obj_class = 1
        elif (df2[3][m+1+offset] == 'Platelets'): obj_class = 2 # 0: RBC, 1: WBC, 2: Platelets

        x,y= (int(df2[6][m+1+offset])+int(df2[4][m+1+offset]))/2,(int(df2[7][m+1+offset])+int(df2[5][m+1+offset]))/2
        bw,bh= int(df2[6][m+1+offset])-int(df2[4][m+1+offset]), int(df2[7][m+1+offset])-int(df2[5][m+1+offset])

        x=int(x); y=int(y)

        i,j=int(x/GW),int(y/GH)

        if ((i >= 11) | (j >= 11)):
            i -= 1
            j -= 1

        test_label_conf[n,j,i] = 1
        test_label_coord[n,j,i,0:2]=((x-i*GW)/GW),((y-j*GH)/GH) # Normalized
        test_label_size[n,j,i,0:2]=(bw/(W/4)),(bh/(H/4)) # Normalized
        test_label_class[n,j,i,0]= obj_class
    offset += M

# Preprocess test images
test_img_ = tf.keras.applications.inception_v3.preprocess_input(test_img)

# Predict object locations in test images
model=tf.keras.models.load_model('model',
                                custom_objects={'conf_loss_func':conf_loss_func,
                                                'coord_loss_func':coord_loss_func,
                                                'size_loss_func':size_loss_func,
                                                'class_loss_func':class_loss_func})

pre_conf,pred_coord,pred_size,pred_class=model.predict(test_img_)

#Display
for n in range(N):
    for i in range(GWN):
        for j in range(GHN):
            if pre_conf[n,j,i]>0.4:
                x=int(i*GW+(pred_coord[n,j,i,0]*GW))
                y=int(j*GH+(pred_coord[n,j,i,1]*GH))

                bw=int(pred_size[n,j,i,0]*(W/4))
                bh=int(pred_size[n,j,i,1]*(H/4))

                obj_class=np.argmax(pred_class[n,j,i,:],axis=-1)

                if obj_class==0: # RBC, Red
                    cv.rectangle(test_img[n], (x - int(bw / 2), y - int(bh / 2)), (x + int(bw / 2), y + int(bh / 2)),
                                 color=(0, 0, 255), thickness= 2)
                    cv.circle(test_img[n], center=(int(i*GW+(test_label_coord[n,j,i,0]*GW)),int(j*GH+(test_label_coord[n,j,i,1]*GH))), radius=40, color=(0, 0, 255), thickness=1)

                elif obj_class==1: # WBC, Blue
                    cv.rectangle(test_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)),
                    color=(255,0,0),thickness= 2)
                    cv.circle(test_img[n], center=(int(i * GW + (test_label_coord[n, j, i, 0] * GW)),
                                                   int(j * GH + (test_label_coord[n, j, i, 1] * GH))), radius=40,
                              color=(255, 0, 0), thickness=1)

                else: # Platelets, Green
                    cv.rectangle(test_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)),
                    color=(0,255,0),thickness= 2)
                    cv.circle(test_img[n], center=(int(i * GW + (test_label_coord[n, j, i, 0] * GW)),
                                                   int(j * GH + (test_label_coord[n, j, i, 1] * GH))), radius=40,
                              color=(0, 255, 0), thickness=1)


    cv.imshow("test_img", test_img[n])
    cv.waitKey(0)
    cv.destroyWindow("test_img")
