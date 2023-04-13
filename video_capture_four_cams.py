import cv2
import time

# Open a connection to the cameras
cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap3 = cv2.VideoCapture(3, cv2.CAP_DSHOW)
caps = [cap0, cap1, cap2, cap3]

sizes = []

for cap in caps:
    width = int(cap.get(3))//2
    height = int(cap.get(4))//2
    size = (width, height)
    cap.set(3, width)
    cap.set(4, height)
    sizes.append(size)
    
# Compression
lossless = cv2.VideoWriter_fourcc(* 'MJPG')

recording = False

while True:

    # Capture a frame from the cameras
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    if ret0 and ret1 and ret2 and ret3:

        frames = [frame0, frame1, frame2, frame3]
        
        for i, frame in enumerate(frames):
            cv2.imshow("Webcam " + str(i), frame)
            
        # Check if the user pressed the enter key
        pressedKey = cv2.waitKey(1) & 0xFF
        
        if pressedKey == cv2.ord('s') :

            print("'s' a été appuyé pour lancer l'enregistrement")

            # Save the frames as video files
            base_name = str(int(time.time()))
            file_names = [base_name + '_cam'+str(i) +'.avi' for i in range(len(caps))]
            videos = [cv2.VideoWriter(file_name, lossless, 30, size) for (file_name, size) in zip(file_names, sizes)]
            recording = True

        if recording:
            for video, frame in zip(videos, frames):
                video.write(frame)
         
        if pressedKey == ord('d'):

            print("'d' a été appuyé pour arrêter la vidéo")
            recording = False
            for video in videos:
                video.release()

        elif pressedKey == 'q':
            print("'q' a été appuyé pour quitter")
            break
    

# Release the camera and close the window
for cap in caps:
    cap.release()

cv2.destroyAllWindows()