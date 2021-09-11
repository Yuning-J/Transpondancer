
# do the necesseary imports
import cv2
import time

# load the video
cap = cv2.VideoCapture('../../Figures/AlphaPose_video1.avi')

if not cap.isOpened():
        print("Cannot open video")
        exit()

# size of frames to save the video later on
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter('Prototype.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

while True:
    ret, frame = cap.read()
    timer = time.time()

    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    
    # Get the fps
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    new_fps = f'FPS: ' + str(fps)

    cv2.putText(frame, new_fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

    # print(timer)
    # if timer == 1:
    #     cv2.putText(frame, 'Movement: Arabesque', (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
    cframe = cap.get(cv2.CAP_PROP_POS_FRAMES) # retrieves the current frame number
    timer = (cframe / fps)

    # write the movement on screen
    if 2.5 <= timer <= 4:
        cv2.putText(frame, 'Current Movement: Pirouette ', (1350, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Current Movement: Classifying...', (1200, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    # save the frames 
    if timer <= 5:
        result.write(frame)

    cv2.imshow('Frame', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
            
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()