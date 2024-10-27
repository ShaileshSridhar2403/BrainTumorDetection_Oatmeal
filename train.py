from dataset_preparation import create_yolov8_dataset
from ultralytics import YOLO


if __name__ == '__main__':
    coco_file = '/Users/shaileshsridhar/Documents/Projects/BrainTumorDetectionOatmeal/BrainTumorDetection_Oatmeal/data/braintumor_sampleset/train/_annotations.coco.json'
    image_dir = '/Users/shaileshsridhar/Documents/Projects/BrainTumorDetectionOatmeal/BrainTumorDetection_Oatmeal/data/braintumor_sampleset/train'
    output_dir = './yolo_dataset'
    dataset_yaml = create_yolov8_dataset(coco_file, image_dir, output_dir)

    # Initialize and train model
    model = YOLO('yolov8n.pt')  # nano model
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        imgsz=640,
        epochs=3,
        batch=64,
        patience=1,
        device='cpu',  # use '0,1,2,3' for multiple GPUs
        project='brain_tumor_detection',
        name='yolov8n_tumor',
        pretrained=True,
        optimizer='auto',  # or 'SGD', 'Adam', etc.
        verbose=True,
        seed=42,
        deterministic=True
    )

    # Validate the model
    # metrics = model.val()

    # # Save the model
    # model.save('best_model.pt')