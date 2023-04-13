import cv2
import time
import argparse

# Create the parser
my_parser = argparse.ArgumentParser()

# Add the arguments
my_parser.add_argument('--cameras', type=int, default=4,
                       help='le nombre de cameras', required=False)

my_parser.add_argument('--resolution', type=int, default=2,
                       help='par combien on divise la hauteur et largeur de la resolution des cameras')

# Execute the parse_args() method
args = my_parser.parse_args()
n = args.cameras

# Open a connection to the cameras
caps = []
sizes = []

for i in range(n):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    width = int(cap.get(3)//args.resolution)
    height = int(cap.get(4)//args.resolution)
    size = (width, height)
    cap.set(3, width)
    cap.set(4, height)
    caps.append(cap)
    sizes.append(size)
    
# Compression
lossless = cv2.VideoWriter_fourcc(*'MJPG')

recording = False

while True:

    # Capture a frame from the cameras
    rets = []
    frames = []
    for i in range(n):
        ret, frame = caps[i].read()
        rets.append(ret)
        frames.append(frame)
        
    # Check if the user pressed the enter key
    pressedKey = cv2.waitKey(100) & 0xFF
        
    if all(rets):
        
        for i, frame in enumerate(frames):
            cv2.imshow("Webcam " + str(i), frame)
        
        if pressedKey == ord('s') :

            print("'s' a été appuyé pour lancer l'enregistrement")

            # Save the frames as video files
            base_name = str(int(time.time()))
            file_names = [base_name + '_cam'+str(i) +'.avi' for i in range(len(caps))]
            videos = [cv2.VideoWriter(file_name, lossless, 10, size) for (file_name, size) in zip(file_names, sizes)]
            recording = True

        if recording:
            for video, frame in zip(videos, frames):
                video.write(frame)
         
        if pressedKey == ord('d'):

            print("'d' a été appuyé pour arrêter la vidéo")
            recording = False
            for video in videos:
                video.release()

        elif pressedKey == ord('q'):
            print("'q' a été appuyé pour quitter")
            break
    

# Release the camera and close the window
for cap in caps:
    cap.release()

cv2.destroyAllWindows()