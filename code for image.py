import cv2
from ultralytics import YOLO

# Load YOLOv8 model (downloads first time)
model = YOLO("yolov8n.pt")

# Load image
frame = cv2.imread(r"C:\Users\pooja\smart traffic\traffic.png")

if frame is None:
    print("Error: Image not found")
    exit()

h, w, _ = frame.shape

# Split into 4 roads
roads = [
    frame[0:h//2, 0:w//2],       # Road 1
    frame[0:h//2, w//2:w],       # Road 2
    frame[h//2:h, 0:w//2],       # Road 3
    frame[h//2:h, w//2:w]        # Road 4
]

# Vehicle classes we care about
vehicle_classes = ["car", "motorcycle", "bus", "truck"]

counts = []

# Process each road
for i, road in enumerate(roads):
    results = model(road)

    count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in vehicle_classes:
                count += 1

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(road, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(road, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    counts.append(count)

    # Show count on each road
    cv2.putText(road, f"Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow(f"Road {i+1}", road)

# 🚦 Decision logic
max_road = counts.index(max(counts)) + 1

print("Vehicle count per road:", counts)
print("🚦 Give GREEN signal to Road", max_road)

cv2.waitKey(0)
cv2.destroyAllWindows()
