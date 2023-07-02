#----Importing Dependencies----:

import numpy as np
import mediapipe as mp
import cv2
mp_drawing = mp.solutions.drawing_utils # For Visualizatons of poses...
mp_pose = mp.solutions.pose

#----Setting MediaPipe Instance----:

# Function for calculating angles
def calculate_angle(a,b,c):
    a = np.array(a) #FIRST:
    b = np.array(b) #MID:
    c = np.array(c) #END:
    # formation of radians with the use of x,y co-ordinates of SHOULDER,ELBOW and WRIST:-
    radian = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radian*180.0/np.pi) 
    
    if angle>180.0:
        angle = 360-angle
    return angle

cap = cv2.VideoCapture(0) # 0 is for my laptop's web-cam feed....

# CURL COUNTER FOR BOTH HANDS:-
counter = 0
stage = None

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    # If you want more accurate model detection and tracking, bump this metrix up...
    while cap.isOpened():
        ret, frame = cap.read()
        
        #RECOLORING THE IMAGE TO RGB:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Making Detection:
        results = pose.process(image)
        
        # RECOLORING BACK TO BGR:
        image.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #print(results)
        
        # Extracting Landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Getting co-ordinated:
            shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x , landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x , landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x , landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            #Calculating Angles:
            angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
            angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
            
            #Visualize:
            cv2.putText(image, str(angle_left), 
                       tuple(np.multiply(elbow_left, [640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2, cv2.LINE_AA)
                                         
            cv2.putText(image, str(angle_right), 
                       tuple(np.multiply(elbow_right, [640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2, cv2.LINE_AA)

            
            # CURL COUNTER LOGIC:-
            if angle_left > 130 and angle_right > 130:
                stage = "UP"
            if angle_left < 95 and angle_right < 95 and stage == "UP":
                stage = "DOWN"
                counter += 1
                print(counter)
        except:
            pass                                   
        

        # CURL COUNTER VISUALIZATION:-
        cv2.rectangle(image, (0,0), (225,73), (245, 117,16), -1)
        
        #REP COUNTER DATA:-
        cv2.putText(image, 'PUSHUPS', (10,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (5,60),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255), 1, cv2.LINE_AA)
        
        #STAGE DATA VISUALIZATION:-
        cv2.putText(image, 'STAGE', (95,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (70,60),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255), 1, cv2.LINE_AA)
        
        
        if counter > 2:
                cv2.putText(image, "PUSHUPS", (255, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
           
        #RENDERING DETECTION:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color = (245,117,66), thickness = 2, circle_radius = 2),
                                 mp_drawing.DrawingSpec(color = (245,66,230), thickness = 2, circle_radius = 2))
                     
        cv2.imshow("MEDIAPIPE FEED", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):    #PRESS "Q" for breaking out of our feed...
            break
    cap.release()
    cv2.destroyAllWindows()
