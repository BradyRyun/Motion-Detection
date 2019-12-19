import time
from cv2 import VideoCapture, cvtColor, COLOR_BGR2GRAY, CAP_DSHOW, GaussianBlur, absdiff
from cv2 import threshold, dilate, findContours, boundingRect, contourArea, imshow, rectangle
from cv2 import RETR_EXTERNAL, waitKey, CHAIN_APPROX_SIMPLE, destroyAllWindows, THRESH_BINARY
from datetime import datetime
from pandas import DataFrame

# Picks the first video capture device that the computer uses (adjust the 0 if it is in a different order)
video = VideoCapture(0, CAP_DSHOW)

# initalizing first frame for reference, frame counter, status list, duration list, total time, and data frame.
first_frame = None
frames = 0
status_list=[None, None]
times=[]
start_time = time.time()
df = DataFrame(columns = ["Start", "End", "Duration"])

while True:
    frames = frames+1
    check, frame = video.read()
    
    status = 0
    gray = cvtColor(frame, COLOR_BGR2GRAY)
    gray = GaussianBlur(gray, (21,21), 0)
    
    if first_frame is None:
        first_frame=gray
        # Restarts the while loop.
        continue
    
    # Finding the difference between the original frame and the next frames
    delta_frame = absdiff(first_frame, gray)
    # Setting the difference between the original frame and the changed frame.
    delta_thresh = threshold(delta_frame, 15, 255, THRESH_BINARY)[1]
    # Smoothing the threshold frame
    delta_thresh = dilate(delta_thresh, None, iterations = 3)
    
    # Finds contours on the screen.
    (cnts,_) = findContours(delta_thresh.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    
    for contour in cnts:
        # If the area of the contour is less than 1000 pixels, check the next contour.
        if contourArea(contour) < 10000:
            continue
        status = 1
        # Assigns values from the rectangle bounding the contour.
        (x, y, w, h) = boundingRect(contour)
        #drawing the recatangle around the contours.
        rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
    # Adding a 1 or 0 to the status list.
    status_list.append(status)
    
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
             
    # imshow("Delta Frame", delta_frame)
    # imshow("Gray Frame", gray)
    # imshow("Threshold Frame", delta_thresh)
    imshow("Color Frame", frame)
    key = waitKey(16)
    if key==ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

# Adds the times into the data frame.
for i in range(0, len(times), 2):
    df=df.append({"Start": times[i], "End": times[i+1], "Duration": times[i+1]-times[i]}, ignore_index=True)

# formats a time delta so that I can print it simply.
def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)

total_duration = df['Duration'].sum()
print(strfdelta(total_duration, "Moving objects were in frame for: {seconds} seconds"))
print("Total time of capture elapsed: %s" % (time.time()-start_time))
print("Amount of frames past: " + str(frames))
video.release()
destroyAllWindows()