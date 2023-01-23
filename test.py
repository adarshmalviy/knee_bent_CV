import cv2
import mediapipe as mp
import numpy as np
import timeit


cap1 = cv2.VideoCapture('KneeBendVideo.mp4')
success1, image1 = cap1.read()
fps = 12
height, width, layers = image1.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# parameters
tt =0
du = 0

# Use MediaPipe to detect poses in the video
with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    cap = cv2.VideoCapture('KneeBendVideo.mp4')
    
    counter = 0
    stage = None
    warn = ''

    # Function to calculate the angle between three points
    def calculate_angle(a, b, c):
        a = np.array(a)  # First point
        b = np.array(b)  # Second point
        c = np.array(c)  # Third point

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180:
            angle = 360 - angle

        return angle
    
    start_time = timeit.default_timer()
    buffer = []
    buffer_size = 10
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Frame ended")
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate angle
            angle = calculate_angle(hip, knee, ankle)

            # Visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # Solving Fluctuation problem using buffer
            buffer.append(angle)
            if len(buffer) > buffer_size:
                buffer.pop(0)
            angle = sum(buffer) / len(buffer)

            # Adding holding time of 8 second and counting rep
            if angle <= 140:
                if stage != "bent":
                    start_time = timeit.default_timer()
                    tt = start_time
                stage = "bent"
            elif angle >= 160 and stage == "bent":
                duration = timeit.default_timer() - start_time
                du = duration
                if duration >= 8.0:
                    stage = "straight"
                    counter += 1
                elif duration < 8.0:
                    warn = "Keep your knee bent"

        except:
            pass

        # feedback
        cv2.putText(image, "Flag:", (image.shape[1]-200+50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, warn, (image.shape[1]-180+60, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Rep
        cv2.putText(image, "REPs: ", (image.shape[1]-200+50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, str(counter), (image.shape[1]-110+50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Start Time & Duration
        cv2.putText(image, "start time:"+str(round(tt,2))+" sec", (image.shape[1]-200+50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "Duration:"+str(round(du,2))+" sec", (image.shape[1]-200+50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Knee Bend Tracker', image)
        
        # Write the processed frame to the output video
        out.write(image)  

        print(counter)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

