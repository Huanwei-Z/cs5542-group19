
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array
#调用笔记本内置摄像头，参数为0，如果有其他的摄像头可以调整参数为1,2
cap = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(r'C:\Users\Administrator\AppData\Local\Programs\Python\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')  #待更改
#为即将录入的脸标记一个id
#sampleNum用来计数样本数目
count = 0
height=250
width=250
def image_array(image):
    image = cv2.resize(image, (height, width))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image
    
model = load_model('./model/classify.model')
while True:
    #从摄像头读取图片
    success,img = cap.read()
    image = image_array(img)
    result = model.predict(image)[0]
    print(result)
    if success is True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        break
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0))
        count += 1
    cv2.imshow('image',img)
        #保持画面的连续。waitkey方法可以绑定按键保证画面的收放，通过q键退出摄像
    print(list(result).index(max(list(result))))
    
    k = cv2.waitKey(10)
    if k & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
