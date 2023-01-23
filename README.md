# knee_bent_CV


1.	Make sure that you have the necessary dependencies installed: cv2, MediaPipe, NumPy, and timeit.

2.	Run the script using the python interpreter: python script_name.py

3.	The script uses the OpenCV library to read a video file called "KneeBendVideo.mp4" and the MediaPipe library to detect key points in the video.

4.	The script then uses these key points to calculate the angle of the knee joint and determine whether the knee is in a bent or straight position.

5.	A buffer is used to smooth out rapid fluctuations in the angle calculation.

6.	The script starts a timer when the knee is in a bent position. When the knee goes back to a straight position, the code will check if the timer has been running for at least 8 seconds. If so, it will increment the rep count and reset the timer. If not, it will display the feedback message "Keep your knee bent" on the video

7.	The script will display the number of reps completed and will display a warning message if the user fails to hold the knee bend position for at least 8 seconds.

8.	The user can exit the program by pressing the 'q' key.

9.	To adjust the buffer size, you can change the buffer size variable. A larger buffer size will smooth out more fluctuations but may also cause the algorithm to be less responsive to rapid changes. A smaller buffer size will be more responsive to rapid changes but may also be more affected by fluctuations.

10.	If the person is not detected in the video, the script will continue to run but will not display any information.
