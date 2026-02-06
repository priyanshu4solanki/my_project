from ultralytics import YOLO

# 1️⃣ Load a base YOLOv8 model
model = YOLO("yolov8n.pt")

# 2️⃣ Start training
model.train(
    data=r"C:\Users\priyanshu solanki\Downloads\Weapon Detection using YOLOv8.v1i.yolov8\data.yaml.yaml", 
    # <‑ full path to your YAML
    epochs=20,
    imgsz=640,
    batch=16,
    project="runs/train",
    name="weapon_detection"
)
print("✅ Training complete! Check 'runs/train/weapon_detection/weights/best.pt'")
