import cv2

cap = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    road1 = frame[0:h//2,  0:w//2]
    road2 = frame[0:h//2,  w//2:w]
    road3 = frame[h//2:h,  0:w//2]
    road4 = frame[h//2:h,  w//2:w]

    cv2.imshow("Road 1", road1)
    cv2.imshow("Road 2", road2)
    cv2.imshow("Road 3", road3)
    cv2.imshow("Road 4", road4)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()