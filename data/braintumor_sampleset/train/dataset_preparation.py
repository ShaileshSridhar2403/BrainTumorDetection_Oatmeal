from ultralytics import YOLO
import json
import os
from sklearn.model_selection import train_test_split
import yaml
import shutil

def create_yolov8_dataset(coco_file, image_dir, output_dir):
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Create category id to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Prepare image paths and labels
    image_paths = []
    labels = []

    for image in coco_data['images']:
        image_id = image['id']
        image_path = os.path.join(image_dir, image['file_name'])
        image_paths.append(image_path)

        image_labels = []
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id:
                cat_id = ann['category_id']
                x, y, w, h = ann['bbox']
                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center = (x + w / 2) / image['width']
                y_center = (y + h / 2) / image['height']
                w = w / image['width']
                h = h / image['height']
                image_labels.append(f"{cat_id} {x_center} {y_center} {w} {h}")

        labels.append('\n'.join(image_labels))

    # Split data into train and val sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)

    # Write data to files
    for images, labels, split in [(train_images, train_labels, 'train'), (val_images, val_labels, 'val')]:
        for img_path, label in zip(images, labels):
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            
            # Copy image
            shutil.copy(img_path, os.path.join(output_dir, 'images', split, img_name))
            
            # Write label
            with open(os.path.join(output_dir, 'labels', split, label_name), 'w') as f:
                f.write(label)

    # Create dataset.yaml file
    dataset_yaml = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(cat_id_to_name),
        'names': list(cat_id_to_name.values())
    }

    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)

    return yaml_path

def main():
    coco_file = 'path/to/your/coco_annotations.json'
    image_dir = 'path/to/your/image/directory'
    output_dir = 'path/to/output/yolov8/dataset'

    # Prepare dataset
    dataset_yaml = create_yolov8_dataset(coco_file, image_dir, output_dir)

    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # 'n' for nano, you can use 's', 'm', 'l', or 'x' for larger models

    # Train YOLOv8 model
    results = model.train(
        data=dataset_yaml,
        imgsz=640,
        batch=16,
        epochs=100,
        patience=50,
        project='tumor_detection',
        name='yolov8n_tumor'
    )

    # Evaluate the model
    results = model.val()

    # Perform inference on a test image
    results = model('path/to/test/image.jpg')

    # Export the model to ONNX format
    success = model.export(format='onnx')

if __name__ == '__main__':
    main()