from ultralytics import YOLO
import os
from tide import analyze_model_predictions

def evaluate_model(
    model_path,
    dataset_yaml,
    val_data_path,
    coco_file,
    output_dir='evaluation_results',
    conf_threshold=0.25,
    imgsz=224,
    batch=32,
    device='cpu'
):
    """
    Evaluate a trained YOLOv8 model on validation data
    
    Args:
        model_path: Path to the saved model weights
        dataset_yaml: Path to dataset YAML file
        val_data_path: Path to validation data directory
        coco_file: Path to COCO annotation file
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for predictions
        imgsz: Image size
        batch: Batch size for validation
        device: Device to run evaluation on ('cpu' or 'cuda')
    
    Returns:
        metrics: Validation metrics including mAP, precision, and recall
    """
    # Load the model
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run validation
    metrics = model.val(
        data=dataset_yaml,
        device=device,
        batch=batch,
        imgsz=imgsz,
        project=output_dir,
        name='validation_results',
        verbose=True
    )
    
    # Print validation metrics
    print("\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
    
    # Run TIDE analysis
    analyze_model_predictions(
        model=model,
        val_data=val_data_path,
        coco_file=coco_file,
        output_dir=output_dir,
        conf_threshold=conf_threshold,
        imgsz=imgsz
    )
    
    return metrics

if __name__ == '__main__':
    # Example usage
    metrics = evaluate_model(
        model_path='/Users/shaileshsridhar/Documents/Projects/BrainTumorDetectionOatmeal/BrainTumorDetection_Oatmeal/brain_tumor_detection/yolov8n_tumor9/weights/best.pt',
        dataset_yaml='yolo_dataset/dataset.yaml',
        val_data_path='yolo_dataset/images/val',
        coco_file='data/braintumor_sampleset/train/_annotations.coco.json',
        output_dir='evaluation_results',
        conf_threshold=0.25,
        imgsz=224,
        batch=32,
        device='cpu'  # Change to 'cuda' if using GPU
    )
