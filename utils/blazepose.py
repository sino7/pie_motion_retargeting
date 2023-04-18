import cv2
import mediapipe as mp
import torch
mp_pose = mp.solutions.pose


def skeleton_from_blazepose_landmarks(landmarks):
    landmarks = torch.Tensor([
        [lm.x, lm.y, lm.z] for lm in landmarks
    ])
    skeleton = torch.zeros(17, 3)
    skeleton[0] = (landmarks[23] + landmarks[24]) / 2
    skeleton[1] = landmarks[23]
    skeleton[2] = landmarks[25]
    skeleton[3] = landmarks[27]
    skeleton[4] = landmarks[24]
    skeleton[5] = landmarks[26]
    skeleton[6] = landmarks[28]
    skeleton[8] = (landmarks[11] + landmarks[12]) / 2
    skeleton[7] = (skeleton[0] + skeleton[8]) /2
    skeleton[9] = (skeleton[8] + landmarks[0]) /2
    skeleton[10] = landmarks[0]
    skeleton[11] = landmarks[11]
    skeleton[12] = landmarks[13]
    skeleton[13] = landmarks[15]
    skeleton[14] = landmarks[12]
    skeleton[15] = landmarks[14]
    skeleton[16] = landmarks[16]
   
    return skeleton

def blazepose_skeletons(video_file, split=None):

    skeletons = []

    cap = cv2.VideoCapture(video_file)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
                      
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks is not None:
                frame_skeleton = skeleton_from_blazepose_landmarks(results.pose_landmarks.landmark)
                skeletons.append(frame_skeleton)

    cap.release()
    cv2.destroyAllWindows()
   
    return torch.cat([sk.unsqueeze(0) for sk in skeletons], axis=0)