import os
import sys
import cv2
import numpy as np
from imutils import paths
from keras.preprocessing.image import img_to_array

from sklearn import svm
from sklearn.decomposition import PCA


height=250
width=250
classes = 67
epochs=1
batch_size=3
save_path='model/0/'


#Get the image data, the image size is uniform
def getimage(path):
    lst_img = []
    lst_label = []
    i=0
    for label in os.listdir(path):
        images = paths.list_images(path+label)
        for img in images:
            image = cv2.imdecode(np.fromfile(img,dtype=np.uint8),-1)
            i=i+1
            if image is None:
                os.system('rm -rf %s' % img)
                continue
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            try:
                lst=[]
                image = cv2.resize(image, (height, width), interpolation = cv2.INTER_AREA)
                image = img_to_array(image)
                for i in range(len(image)):
                    for j in range(len(image[i])):
                        lst.append(image[i][j][0])
            
                lst_img.append(lst)
                lst_label.append(int(label))
#                image = img_to_array(image)
            except:
                print(img)
    np_img = np.array(lst_img) 
    print(np_img)
    np_label = np.array(lst_label)
    return np_img, np_label


def getimage2(path):
    lst_img = []
    lst_label = []
    lst_path= []
    i=0
    for label in os.listdir(path):
        images = paths.list_images(path+label)
        for img in images:
            image = cv2.imdecode(np.fromfile(img,dtype=np.uint8),-1)
            i=i+1
            if image is None:
                os.system('rm -rf %s' % img)
                continue
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            try:
                lst=[]
                image = cv2.resize(image, (height, width), interpolation = cv2.INTER_AREA)
                image = img_to_array(image)
                for i in range(len(image)):
                    for j in range(len(image[i])):
                        lst.append(image[i][j][0])
                
                lst_path.append(img)
                lst_img.append(lst)
                lst_label.append(int(label))
#                image = img_to_array(image)
            except:
                print(img)
    np_img = np.array(lst_img) 
    print(np_img)
    np_label = np.array(lst_label)
    return np_img, np_label, lst_path


#Training model
def train_model(train_path, test_path):
    model = svm.SVC(gamma='auto')
    train_img, train_label = getimage(train_path)
    test_img, test_label, lst_path = getimage2(test_path)
    
    pca = PCA().fit(train_img)
    
    train_img = pca.transform(train_img)
    test_img = pca.transform(test_img)
    
    model.fit(train_img, train_label)
    
    y_pred = model.predict(test_img)
    print(y_pred)
    score = model.score(test_img, test_label)#[1,2,3]
    
    file=open('result/svmresult.txt', 'w')
    for i in range(len(y_pred)):
        file.write(('graph%s'%lst_path[i])+('recognized as cow No.%d'%y_pred[i])+'\n')
    file.write('Accuracy:'+str(int(score * 100))+'%')
    file.close()
    
    print("Training completed,the model score is %s" % score)
    print('Training prediction results: ' + str(int(score * 100))+'%')

if __name__ == '__main__':
    train = "./data/train/"
    test = "./data/test/"
    train_model(train, test)
