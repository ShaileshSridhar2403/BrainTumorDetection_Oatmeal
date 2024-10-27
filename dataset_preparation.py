from ultralytics import YOLO
import json
import os
from sklearn.model_selection import train_test_split
import yaml
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_yolov8_dataset(coco_file, image_dir, output_dir, target_size=(224, 224)):
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Create category id to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)

    # Prepare image paths and labels
    image_paths = []
    labels = []

    for image in coco_data['images']:
        image_id = image['id']
        image_path = os.path.join(image_dir, image['file_name'])
        image_paths.append(image_path)

        # Collect all annotations for this image
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

    # Split data
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    # Write data to files
    for images, labels, split in [(train_images, train_labels, 'train'), 
                                (val_images, val_labels, 'val')]:
        for img_path, label in zip(images, labels):
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            
            # Load image and get original dimensions
            img = Image.open(img_path)
            original_width, original_height = img.size
            
            # Determine scaling factors
            width_scale = target_size[0] / original_width
            height_scale = target_size[1] / original_height
            
            # Resize image
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img.save(os.path.join(output_dir, 'images', split, img_name))
            
            # Adjust bounding boxes
            scaled_labels = []
            for label_line in label.split('\n'):
                if label_line.strip():
                    class_id, x_center, y_center, w, h = map(float, label_line.split())
                    
                    # Convert normalized coordinates back to absolute
                    x_center_abs = x_center * original_width
                    y_center_abs = y_center * original_height
                    w_abs = w * original_width
                    h_abs = h * original_height
                    
                    # Scale absolute coordinates
                    x_center_scaled = x_center_abs * width_scale
                    y_center_scaled = y_center_abs * height_scale
                    w_scaled = w_abs * width_scale
                    h_scaled = h_abs * height_scale
                    
                    # Convert back to normalized coordinates
                    x_center_norm = x_center_scaled / target_size[0]
                    y_center_norm = y_center_scaled / target_size[1]
                    w_norm = w_scaled / target_size[0]
                    h_norm = h_scaled / target_size[1]
                    
                    # Ensure values are within [0, 1]
                    x_center_norm = min(max(x_center_norm, 0), 1)
                    y_center_norm = min(max(y_center_norm, 0), 1)
                    w_norm = min(max(w_norm, 0), 1)
                    h_norm = min(max(h_norm, 0), 1)
                    
                    scaled_labels.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            # Write label
            with open(os.path.join(output_dir, 'labels', split, label_name), 'w') as f:
                f.write('\n'.join(scaled_labels))

    # Create dataset.yaml
    dataset_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(cat_id_to_name),
        'names': list(cat_id_to_name.values())
    }

    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    return yaml_path

def visualize_yolo_annotations(image_path, label_path):
    """
    Visualize YOLO format annotations (after conversion)
    """
    # Load image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Load YOLO format annotations
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Get image dimensions
    img_w, img_h = image.size
    
    # Plot each bounding box
    for line in lines:
        class_id, x_center, y_center, w, h = map(float, line.strip().split())
        
        # Convert normalized YOLO format to pixel coordinates
        x = (x_center - w/2) * img_w
        y = (y_center - h/2) * img_h
        w = w * img_w
        h = h * img_h
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(x, y-5, f'Class {int(class_id)}', color='red', fontsize=8,
               bbox=dict(facecolor='white', alpha=0.7))
    
    ax.axis('off')
    plt.show()

def verify_conversion(output_dir, num_samples=5):
    """
    Visualize YOLO format annotations with all images in a row
    """
    train_dir = os.path.join(output_dir, 'images', 'train')
    label_dir = os.path.join(output_dir, 'labels', 'train')
    
    # Randomly sample image files
    image_files = random.sample(os.listdir(train_dir), num_samples)
    
    # Create subplot grid with more space for title
    plt.figure(figsize=(20, 5))  # Increased height slightly
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    fig.suptitle('Resized Images (224x224)', fontsize=16, y=0.95)  # Lowered y position
    if num_samples == 1:
        axes = [axes]
    
    # Plot each image with its bounding boxes
    for idx, (ax, img_file) in enumerate(zip(axes, image_files), 1):
        image_path = os.path.join(train_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        
        # Load and display image
        image = Image.open(image_path)
        ax.imshow(image)
        
        # Get image dimensions
        img_w, img_h = image.size
        
        # Load and plot YOLO format annotations
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                
                # Convert normalized YOLO format to pixel coordinates
                x = (x_center - w/2) * img_w
                y = (y_center - h/2) * img_h
                w = w * img_w
                h = h * img_h
                
                # Create rectangle patch
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(x, y-5, f'Class {int(class_id)}', color='red', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_title(f'Random Image {idx}')
        ax.axis('off')
    
    plt.subplots_adjust(top=0.85)  # Add space between title and plots
    plt.tight_layout()
    plt.show()

def visualize_sample_images(coco_file, image_dir, num_samples=5):
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to annotations mapping
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Create category id to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Randomly sample images
    sample_images = random.sample(coco_data['images'], num_samples)
    
    # Create subplot grid with more space for title
    plt.figure(figsize=(20, 5))  # Increased height slightly
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    fig.suptitle('Original Images (Before Resizing)', fontsize=16, y=0.95)  # Lowered y position
    if num_samples == 1:
        axes = [axes]
    
    # Plot each image with its bounding boxes
    for idx, (ax, img_data) in enumerate(zip(axes, sample_images), 1):
        # Load and display image
        img_path = os.path.join(image_dir, img_data['file_name'])
        image = Image.open(img_path)
        ax.imshow(image)
        
        # Get annotations for this image
        annotations = image_annotations.get(img_data['id'], [])
        
        # Plot each bounding box
        for ann in annotations:
            x, y, w, h = ann['bbox']
            category = cat_id_to_name[ann['category_id']]
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(x, y-5, category, color='red', fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_title(f'Random Image {idx}')
        ax.axis('off')
    
    plt.subplots_adjust(top=0.85)  # Add space between title and plots
    plt.tight_layout()
    plt.show()

def main():
    # Update these paths to your actual paths
    coco_file = '/Users/shaileshsridhar/Documents/Projects/BrainTumorDetectionOatmeal/BrainTumorDetection_Oatmeal/data/braintumor_sampleset/train/_annotations.coco.json'
    image_dir = '/Users/shaileshsridhar/Documents/Projects/BrainTumorDetectionOatmeal/BrainTumorDetection_Oatmeal/data/braintumor_sampleset/train'
    output_dir = './yolo_dataset'

    print("Visualizing dataset...")
    visualize_sample_images(coco_file, image_dir, num_samples=5)
    
    dataset_yaml = create_yolov8_dataset(coco_file, image_dir, output_dir, target_size=(224, 224))
    
    verify_conversion(output_dir, num_samples=5)
    
    # Continue with training...

if __name__ == '__main__':
    main()
