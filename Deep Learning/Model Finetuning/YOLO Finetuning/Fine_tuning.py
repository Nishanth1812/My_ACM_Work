from ultralytics import YOLO
import cv2 


# Load YOLO model
model = YOLO(r"C:\Users\Devab\OneDrive\Desktop\Coding\ML-DL\Deep Learning\Applications\YOLO Finetuning\best.pt")


# Initialize webcam
web_cam = cv2.VideoCapture(0)

# Set width and height of webcam f
# eed
web_cam.set(3, 640)
web_cam.set(4, 480)

while web_cam.isOpened():
    success, img_frame = web_cam.read()
    if not success:
        break  # Exit if frame not captured properly
    
    # Run YOLO model on the frame
    results = model(img_frame, conf=0.8)
    a_frame = results[0].plot()

    # Display result
    cv2.imshow("Result", a_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and destroy all windows
web_cam.release()
cv2.destroyAllWindows()
