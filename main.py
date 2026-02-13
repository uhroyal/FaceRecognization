import threading
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # width of cam
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #height of cam 

counter = 0
face_match = False

reference_img = cv2.imread("reference.jpg", cv2.IMREAD_GRAYSCALE) #reference image for face recognition

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def check_face(frame): #function to check if face is detected in the frame
    global face_match
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            face_match = True
        else:
            face_match = False
    except Exception:
        face_match = False
# main loop to read frames from the webcam and check for faces every 30 frames
while True: 
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except Exception:
                pass
        counter += 1
# Display the result on the frame
        if face_match:
            cv2.putText(frame, "FACE DETECTED!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO FACE", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Face Recognition", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


# @uhroyal