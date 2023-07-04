import cv2

videoFeed = cv2.VideoCapture(1)
faceCapture = cv2.CascadeClassifier("/home/palash/opencv/data/haarcascades/haarcascade_frontalface_default.xml")
while True:
    isRead, videoData = videoFeed.read()
    color = cv2.cvtColor(videoData,cv2.COLOR_BGR2GRAY)
    faces = faceCapture.detectMultiScale(
        color,
        flags = cv2.CASCADE_SCALE_IMAGE,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize= (30,30)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(videoData, (x,y) , (x+w,y+h),(0,255,0), 2)
    while isRead == False:
        print("Not Done")
    if isRead:
        cv2.imshow("Hello_World!", videoData)
    else:
        print("Error reading video feed.")
        break

    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):
        break

videoFeed.release()
cv2.destroyAllWindows()
