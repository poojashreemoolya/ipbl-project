import cv2

cap = cv2.VideoCapture(r"C:\Users\pooja\smart traffic\traffic.png")

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Split into 4 roads
    roads = [
        frame[0:h//2, 0:w//2],
        frame[0:h//2, w//2:w],
        frame[h//2:h, 0:w//2],
        frame[h//2:h, w//2:w]
    ]

    counts = []

    for i, road in enumerate(roads):
        fgmask = fgbg.apply(road)

        # Remove noise
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                count += 1
                x, y, w1, h1 = cv2.boundingRect(cnt)
                cv2.rectangle(road, (x,y), (x+w1,y+h1), (0,255,0), 2)

        counts.append(count)

        cv2.putText(road, f"Count: {count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow(f"Road {i+1}", road)

    print("Counts:", counts)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()