from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Evaluate model performance on the validation set

# Perform object detection on an image
results = model(r"C:\Users\Devab\OneDrive\Desktop\Coding\ML-DL\Deep Learning\Applications\YOLO Model\images.jpeg")
results[0].show()