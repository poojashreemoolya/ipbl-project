import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Load image
frame = cv2.imread(r"C:\Users\Pallavi\OneDrive\Desktop\junction.jpg.jpeg")

if frame is None:
    print("Error: Image not found")
    exit()

h, w, _ = frame.shape

# Vehicle classes
vehicle_classes = ["car", "motorcycle", "bus", "truck"]

# Road counts
counts = {"Bottom":0, "Top":0, "Left":0, "Right":0}

# Run YOLO on full frame
results = model(frame)

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label in vehicle_classes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 🔥 ROAD DIVISION LOGIC (based on your intersection)
            
            # Bottom road (near camera)
            if cy > int(h * 0.65):
                road = "Bottom"

            # Top road
            elif cy < int(h * 0.35):
                road = "Top"

            # Left road
            elif cx < int(w * 0.35):
                road = "Left"

            # Right road
            else:
                road = "Right"

            counts[road] += 1

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, road, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# 🔥 Draw road boundaries (visual clarity)
cv2.line(frame, (0, int(h*0.35)), (w, int(h*0.35)), (255,0,0), 2)
cv2.line(frame, (0, int(h*0.65)), (w, int(h*0.65)), (255,0,0), 2)

cv2.line(frame, (int(w*0.35), 0), (int(w*0.35), h), (255,0,0), 2)
cv2.line(frame, (int(w*0.65), 0), (int(w*0.65), h), (255,0,0), 2)

# Show counts
y_pos = 40
for road, count in counts.items():
    cv2.putText(frame, f"{road}: {count}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    y_pos += 40

# Decision
max_road = max(counts, key=counts.get)

cv2.putText(frame, f"GREEN: {max_road}", (10, y_pos+20),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

print("Counts:", counts)
print("🚦 GREEN signal to:", max_road)

# Show final output
cv2.imshow("Smart Traffic System", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()