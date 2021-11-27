
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array
#Call the notebook built-in camera, the parameter is 0, if there are other cameras, you can adjust the parameter to 1,2
cap = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(r'F:\Python\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')  #待更改
#Mark an id for the face to be entered
#sampleNum is used to count the number of samples
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
    #read the picture from the camera
    success,img = cap.read()
    image = image_array(img)
    result = model.predict(image)[0]
    print(result)
    print("recognized as cow No.%d"%list(result).index(max(list(result))))
    if success is True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        break
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0))
        count += 1
    cv2.imshow('image',img)
        #Keep the picture continuity. The waitkey method can bind the buttons to ensure that the picture is retracted, and exit the camera by pressing the q key

    
    k = cv2.waitKey(10)
    if k & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
