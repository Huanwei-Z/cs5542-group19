# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
from imutils import paths
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array

from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.optimizers import Adam

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from keras import regularizers
from keras import backend as bk
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization

height=250
width=250
classes = 67
epochs=400
batch_size=50
save_path='model/classify.model'

file_list = []  #Create an empty list

def out_file():
    #file_2 = open_file()
    file = 'result/check.txt'    #Open files that need to be deduplicated
    with open(file, "r") as f:
        file_2 = f.readlines()
        for file in file_2:
            file_list.append(file)
        out_file1 = set(file_list)    #The set() function can automatically filter out duplicate elements
        last_out_file = list(out_file1)
        for out in last_out_file:
            with open('result/result.txt',"a+") as f:   #The file is written to the file after deduplication
                f.write(out)
        print("去重完成"+"\n")

def splitimage(frame, i):
    face_class = cv2.CascadeClassifier(r'C:\Users\Administrator\AppData\Local\Programs\Python\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')  #待更改

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Use the classifier to identify which area is the face
    faceRects = face_class.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
    if len(faceRects) > 0:                 
        for faceRect in faceRects: 
            x, y, w, h = faceRect
            image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
        cv2.imwrite(str(i)+'.jpg', image)

#Get the image data, make the image size uniform
def getimage(path):
    lst_img = []
    lst_label = []
    i=0
    for label in os.listdir(path):
        # print(label)
   #     images = paths.list_images(path+label)
        for fn in os.listdir(path+label):
            img=path+label+'/'+fn
        # print(images)
        # for img in images:
        #     image = cv2.imread(img)
            image = cv2.imdecode(np.fromfile(img,dtype=np.uint8),-1)
            i=i+1
            if image is None:
                os.system('rm -rf %s' % img)
                continue
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                image = cv2.resize(image, (height, width), interpolation = cv2.INTER_AREA)
                image = img_to_array(image)
            except:
                print(img)
            lst_img.append(image)
            lst_label.append(int(label))

    np_img = np.array(lst_img, dtype="float") / 255.0
    np_label = np.array(lst_label)
    np_label = to_categorical(np_label, num_classes=classes)
    return np_img, np_label

def getimage2(path):
    lst_img = []
    lst_label = []
    lst_path= []
    i=0
    for label in os.listdir(path):
        # print(label)
   #     images = paths.list_images(path+label)
        for fn in os.listdir(path+label):
            img=path+label+'/'+fn
            
        # print(images)
        # for img in images:
        #     image = cv2.imread(img)
            image = cv2.imdecode(np.fromfile(img,dtype=np.uint8),-1)
            i=i+1
            if image is None:
                os.system('rm -rf %s' % img)
                continue
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                image = cv2.resize(image, (height, width), interpolation = cv2.INTER_AREA)  #mark
                image = img_to_array(image)
            except:
                print(img)
            lst_path.append(img)
            lst_img.append(image)
            lst_label.append(int(label))

    np_img = np.array(lst_img, dtype="float") / 255.0
    np_label = np.array(lst_label)
    np_label = to_categorical(np_label, num_classes=classes) #3  0  【1，0，0】  【0，1，0】
    return np_img, np_label, lst_path


def buildModel(width, heigth, classes, depth=3):
    model = Sequential()
    if bk.image_data_format() == "channels_first":
        shape = (depth, width, heigth)
    else:
        shape = (width, heigth, depth)
    
    #Convolutional layer
    model.add(Conv2D(20, (3, 3), padding="same", input_shape=shape, name='filter1'))
    model.add(Activation("relu")) #Activate the function layer
    model.add(MaxPooling2D(strides=(2, 2), name="max1"))
    model.add(Dropout(0.5))

    model.add(Conv2D(30, (3, 3), padding="same", name='filter2'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(strides=(2, 2), name="max2"))

    model.add(Conv2D(50, (5, 5), padding="same", name='filter3'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(strides=(2, 2), name="max3"))#Pooling layer
    model.add(Dropout(0.5)) #Dropout layer

    model.add(Flatten())#Flatten layer
    model.add(Dense(250))#Fully connected layer
    model.add(Dropout(0.5))
    model.add(Activation("relu"))

    model.add(Dense(classes))
    model.add(Activation("softmax"))#Classification layer

    return model

#training model
def train_model(train_path, test_path):
    model = buildModel(width=width, heigth=height, classes=classes)
    # Compiling the model requires three parameters, the optimizer, the loss function, and the metrics
    model.compile(optimizer=Adam(lr=1e-3, decay=0.0), loss="categorical_crossentropy", metrics=['accuracy'])
    train_img, train_label = getimage(train_path)
    test_img, test_label, lst_path= getimage2(test_path)
    # print(len(test_img))

    datagen = ImageDataGenerator(
    rotation_range=20,#The angle at which the image rotates randomly when the data is increased (range 0～180)
    width_shift_range=0.2,#The amplitude of the horizontal offset of the picture when the data is promoted
    height_shift_range=0.2,#Same as above, but here is vertical
    horizontal_flip=True) #Whether to perform random horizontal flip
    #If it exists, it can be superimposed
    if os.path.exists('checkpoint.chk'):
       model.load_weights("checkpoint.chk")
    try:
        fit = model.fit_generator(
            datagen.flow(train_img, train_label, batch_size=batch_size), 
            validation_data=(test_img, test_label),
            steps_per_epoch = len(train_img) // epochs,
            epochs=epochs,
            verbose=1,
            )
    except:
        fit = model.fit_generator(
            datagen.flow(train_img, train_label, batch_size=batch_size), 
            validation_data=(test_img, test_label),
            epochs=epochs,
            verbose=1,
            )
    score = model.evaluate(test_img, test_label, batch_size=32)
    y_pred = model.predict(test_img, batch_size=32, )
    
    print("Training completed,the model score is %s" % score)
    print("Training prediction results: %s" % y_pred)
#    print(fit.history)
    lst_result=[]
    for t in y_pred:
        t=list(t)
        lst_result.append(t.index(max(t)))

    print(lst_result)
    
    file=open('result/cnnresult.txt', 'w')
    for i in range(len(lst_result)):
        file.write(('image%s'%lst_path[i])+('recognitied as%dcow'%lst_result[i])+'\n')
    file.write('Accuracy：%s'%str(score[1]))
    file.close()

    file = open('result/check.txt', 'w')
    cwj=1
    for i in range(len(lst_result)):
        file.write(('%s:' % lst_result[i]) + ('%d' % cwj) + '\n')
    file.close()

    out_file()
    
    model.save(save_path)
    plt.figure()
    try:
        plt.plot(fit.history["loss"], label="train_loss")
        plt.plot(fit.history["accuracy"], label="train_acc")
        plt.plot(fit.history["val_loss"], label="val_loss")
        plt.plot(fit.history["val_accuracy"], label="val_acc")
    except:
        plt.plot(fit.history["loss"], label="train_loss")
        plt.plot(fit.history["acc"], label="train_acc")
        plt.plot(fit.history["val_loss"], label="val_loss")
        plt.plot(fit.history["val_acc"], label="val_acc")

    plt.title("Image recognition")
    plt.xlabel("Epochs")
    plt.ylabel("loss/acc")
    plt.legend(loc="lower left")
    plt.savefig("reco")
    
if __name__ == '__main__':
    train = "./data/train/"
    test = "./data/test/"
    train_model(train, test)
