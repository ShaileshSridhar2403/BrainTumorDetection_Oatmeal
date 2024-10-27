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

def create_yolov8_dataset(coco_file, image_dir, output_dir):
    # First, load and split the data
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Split the data into train/val/test
    train_images, train_labels = [], []
    val_images, val_labels = [], []
    test_images, test_labels = [], []
    
    # Create category id to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)

    # Prepare image paths and labels
    for images, labels, split in [(train_images, train_labels, 'train'), 
                                 (val_images, val_labels, 'val'),
                                 (test_images, test_labels, 'test')]:
        for img_path, label in zip(images, labels):
            # Copy image
            img_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(output_dir, 'images', split, img_name))
            
            # Write label (already in YOLO format)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            with open(os.path.join(output_dir, 'labels', split, label_name), 'w') as f:
                f.write(label)

    # Create dataset.yaml
    dataset_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
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
    fig.suptitle('YOLO Format', fontsize=16, y=0.95)  # Lowered y position
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
    fig.suptitle('Original Images (Before Conversion)', fontsize=16, y=0.95)  # Lowered y position
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
    
    dataset_yaml = create_yolov8_dataset(coco_file, image_dir, output_dir)

    print("Verifying conversion...")
    verify_conversion(output_dir, num_samples=5)
    
if __name__ == '__main__':
    main()
