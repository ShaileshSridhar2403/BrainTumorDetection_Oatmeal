import tidecv as tide
import json
import os
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from tidecv import TIDE, Data

def convert_yolo_predictions_to_tide(predictions, image_size):
    """Convert YOLO predictions to TIDE format"""
    tide_predictions = {
        "annotations": [],
        "images": [],
        "categories": [{"id": 0, "name": "tumor"}]  # Adjust categories as needed
    }
    
    # Track unique image IDs
    processed_images = set()
    
    for pred in predictions:
        boxes = pred.boxes
        image_id = Path(pred.path).stem
        
        # Add image info if not already added
        if image_id not in processed_images:
            tide_predictions["images"].append({
                "id": image_id,
                "file_name": f"{image_id}.jpg"  # Adjust extension if needed
            })
            processed_images.add(image_id)
        
        # Add detection results
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            annotation = {
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                "score": confidence
            }
            tide_predictions["annotations"].append(annotation)
    
    return tide_predictions

def run_tide_analysis(ground_truth_path, predictions_path, output_dir):
    """Run TIDE analysis and save results"""
    # Load ground truth and predictions
    with open(ground_truth_path, 'r') as f:
        gt_data = json.load(f)
    with open(predictions_path, 'r') as f:
        pred_data = json.load(f)
    
    # Verify we have predictions
    if len(pred_data["annotations"]) == 0:
        print("Warning: No predictions found! TIDE analysis cannot be performed.")
        return
    
    # Create TIDE data objects
    try:
        gt = Data('ground_truth', gt_data)
        pred = Data('predictions', pred_data)
        
        # Initialize TIDE
        tide_eval = TIDE()
        tide_eval.evaluate(
            gt,           # Ground truth TIDE Data object
            pred,         # Predictions TIDE Data object
            mode='bbox',  # Detection mode
            pos_threshold=0.5   # IoU threshold
        )
        
        # Generate and save detailed analysis
        analysis_dir = os.path.join(output_dir, 'tide_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save main error analysis
        tide_eval.summarize()
        tide_eval.plot_summary(os.path.join(analysis_dir, 'error_summary.pdf'))
        
        # Save detailed breakdowns
        error_types = ['Cls', 'Loc', 'Both', 'Dupe', 'Bkg', 'Miss']
        for error_type in error_types:
            tide_eval.plot_error_breakdown(error_type, os.path.join(analysis_dir, f'{error_type}_breakdown.pdf'))
            
    except Exception as e:
        print(f"Error during TIDE analysis: {str(e)}")
        print("Ground truth format:", gt_data.keys() if gt_data else "None")
        print("Predictions format:", pred_data.keys() if pred_data else "None")

def analyze_model_predictions(model, val_data, coco_file, output_dir, conf_threshold=0.25, imgsz=224):
    """Main function to run complete TIDE analysis on model predictions"""
    # Get predictions on validation set
    predictions = model.predict(
        source=val_data,
        save=True,
        project='brain_tumor_detection',
        name='yolov8n_tumor_predictions',
        conf=conf_threshold
    )

    # Convert predictions to TIDE format
    tide_predictions = convert_yolo_predictions_to_tide(predictions, imgsz)
    
    # Save predictions in COCO format
    pred_save_path = os.path.join(output_dir, 'predictions.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(pred_save_path, 'w') as f:
        json.dump(tide_predictions, f)

    print(f"\nSaved predictions to {pred_save_path}")
    print(f"Number of predictions: {len(tide_predictions['annotations'])}")
    
    # Run TIDE analysis
    run_tide_analysis(
        ground_truth_path=coco_file,
        predictions_path=pred_save_path,
        output_dir=output_dir
    )

    print("\nTIDE Analysis Results:")
    print(f"Check the '{output_dir}/tide_analysis' directory for detailed visualizations")
