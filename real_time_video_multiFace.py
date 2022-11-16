from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml' #SUPER many parameters!
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path) #face detection model
emotion_classifier = load_model(emotion_model_path, compile=False) #pretrained loaded model
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
EMOTIONS = ["😠" ,"🤮","😨", "😁", "😥", "😲","😐"]
EMOTIONS = ['angry \ _ / ','','', 'happy ^ _ ^ ', "sad / _ \ ", 'surprised O o O ','neutral - _ - ']

# emo_filter = np.array([1, 0, 0, 1, 1, 1, 0])

#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

# starting video streaming
cv2.namedWindow('your_face') #plot program window
camera = cv2.VideoCapture(0)
print('Frame width:', int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('Frame count:', int(camera.get(cv2.CAP_PROP_FRAME_COUNT)))

# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas_s = []
    for _ in faces:
        canvas_s.append(np.zeros((250, 300, 3), dtype="uint8"))
    frameClone = frame.copy()
    
    if len(faces) > 0:
        '''faces 에는 얼굴의 죄표가 담긴 배열이 저장되어 있음!'''
        '''original code'''
        # faces = sorted(faces, reverse=True,
        # # key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0] #face를 sort하고 첫번째 face를 faces에 넣어줌!
        # (fX, fY, fW, fH) = faces[0] 
        '''face_detection 동시성 다중인식 확인!'''
        # '''multiface code'''
        # for idx, face in enumerate(faces):
        #     print('face{}'.format(idx))
        #     print(face)


        # print(faces) [124 120 102 102]
            # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN

        faces = sorted(faces, key=lambda x: x[0]) #face를 sort함.
        rois = [] #face 를 sort한 이후 관심영역을 폼에 맞춘 형태를 저장!
        for face in faces:

            (fX, fY, fW, fH) = face
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            rois.append(roi)
        '''rois에 변환된 roi 저장!'''

        preds_s = []
        emotion_probability_s = []
        label_s = []
        for idx, roi in enumerate(rois):
            preds = emotion_classifier.predict(roi)[0] 
            # print(preds)
            # preds = emotion_classifier.predict(roi)[0] * emo_filter
            preds_s.append(emotion_classifier.predict(roi)[0])
            
            emotion_probability = np.max(preds)
            emotion_probability_s.append(emotion_probability)
            
            if preds.argmax() == 1 or preds.argmax() == 2:
                label = EMOTIONS[4]
            else:
                label = EMOTIONS[preds.argmax()]
            label_s.append(label)
    else: continue

    '''preds_s까지는 잘 나옴!!!!
    동시에 rectangle 두개 씌우는게 안됨..!!'''
    # print(len(preds_s))

    for idx, preds in enumerate(preds_s):
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                    # draw the label + probability bar on the canvas
                # emoji_face = feelings_faces[np.argmax(preds)]

                    
                    # w = int(prob * 300)
                    # cv2.rectangle(canvas_s[idx], (7, (i * 35) + 5),
                    #     (w, (i * 35) + 35), (0, 0, 255), -1) #7개의 bar graph 생성
                    # cv2.putText(canvas_s[idx], text, (10, (i * 35) + 23),
                    #     cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    #     (255, 255, 255), 2)
                    
                    (fX, fY, fW, fH) = faces[idx]
                    cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                (0, 0, 255), 2)
    #    for c in range(0, 3):
#        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
#        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
#        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)


    cv2.imshow('your_face', frameClone)
    # for idx, canvas in enumerate(canvas_s):
    #     cv2.imshow("Probabilities{}".format(idx), canvas)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
