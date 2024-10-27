
# Process and Notes:

First I check out the images and think about the task. This is object detection for brain tumors. We have an adequate number of training images (~3000) so nothing to worry about there. I do a brief literature survey to see what rough approach is known to work well.

Links:
https://www.mdpi.com/2076-3417/13/16/9158 - yolov8 for brain tumor detection

[https://www.nature.com/articles/s41598-024-57970-7 - deep learning for brain tumor detection](https://arxiv.org/pdf/2307.16412) - another paper using yolo (custom architecture)

It seems that YOLO architectures are known to perform well on this task. Using Ultralytics' implementation, YOLOv8 is easy to set up and has a "nano" sized model which can be trained fast. We decide to use this for our task.



## Dataset:

It looks like there are 2 categories/classes:

1. "Tumor" (with id: 0)
    1. "0" (with id: 1, supercategory: "Tumor")
    2. "1" (with id: 2, supercategory: "Tumor")

For now we will treat these as two separate classes in the interest of time as later we can easily convert back to one class if necessary in the application layer but it is better to understand their significance

We visualize the dateset and we verify that the data and bounding boxes seem fine. (It is however noteworthy that the bounding boxes are not very tight and there is a lot of space between the tumor and the edge of the bounding box, so if given more time we might want to take this into account)

We also verify the conversion to YOLO format. 

All looks good! Time to train a basic model

## Model Training

We use a pretrained model (MS COCO dataset) and train it on our dataset. We use basic augmentations during training, defaults for the ultralytics library, such as blurring, median blurring, and CLAHE.



Each epoch seems to take ~13 minutes on our CPU which is simply too long for this 2 hour task. We do things:

1. Reduce number of epochs to 10
2. Increase batch size to 64 (Taking advantage of the fact that my latop has high ram)
3. Reduce image size to 224x224 (from 640x640)
4. Switch to Google Colaboratory's backend for its GPU

These changes make each epoch take 10 seconds to train which allows us to train and validate, as well as adjust hyperparameters

Important Note: Reducing the image size makes training much faster but also reduces accuracy and might be perceived as changing the task. I use this to quickly validate my approach and then revert it back to 640x640 for a final run


Ultralytics is awesome in the way they provide metrics and outputs after training. We do not need to write code for this explicitly.

After validating, and looking at the result, our results are reasonable. We are predicting several tumors correctly. We also have false positives.

In order to improve the model there are several things we could do:

1. Data Augmentation
2. Better Hyperparameters Tuning
3. Simply train for more epochs
4. Analyze the metrics and see where we can improve (With metrics such as TIDE)


1. is the easiest to do.

We perform ablations with several different augmentations:

* Vertical and horizontal flips - This task seems agnostic to the vertical or horizontal orientation so this will help introducde more data
* Mixup - This will also help the model generalize better

Interestingly, in our example we do not see immediate results with these augmentation strategies, but this is as expected. Augmentation improves generalization in the long run, affecting convergence rather than performance with a smaller number of epochs and hence we choose to keep it.

We notice that the loss is still decreasing from the loss curve. Hence it is a good idea to train for more epochs.
![image](https://github.com/user-attachments/assets/25eb16f7-a311-4a7c-8bb8-1f42f0961106)


In our final run we increase the number of epochs to 15, use flip and mixup augmentations, scale the image size back to 640x640 and run. Ideally we would like to run for more epochs, until the curve plateaus, but due to time constraints we stop at 15.

## Results

                    Class     Images  Instances      P          R      mAP50  mAP50-95
                   all        301        302      0.772      0.766      0.815      0.485
                     0        163        163      0.731      0.632       0.71      0.391
                     1        138        139      0.813      0.899      0.919       0.58



We get reasonable results, with an mAP of 0.919 at a threshold of 0.5 as well as reasonable precision and recall, both above 0.75

### Precision Recall Curve

<img src="https://github.com/user-attachments/assets/e311df76-72ac-4a57-8c33-1b6bc0e30d2e" width="600">


### True Labels
<img src="https://github.com/user-attachments/assets/24208708-07f1-4fbe-bd5f-df563aa49712" width="600">

### Predicted Labels with confidence
<img src="https://github.com/user-attachments/assets/18836623-58b9-4125-addc-78df9fe2db6c" width="600">




We still see a couple of wrong detections. From this random batch, false negatives seem to be a big problem, although a good amount of this will go away with more training.

I also tried to use TIDE toolbox(https://dbolya.github.io/tide/paper.pdf) to look a little deeper at the types of errors our model makes but was unable to debug it and get it working within time constraints. The code is attached.

Taking these further steps, as well as addressing the slight disparity between the two classes, we can see that we can further improve the model
    


## References and Resources Used:

https://www.mdpi.com/2076-3417/13/16/9158 - YOLOv8 for brain tumor detection

[https://www.nature.com/articles/s41598-024-57970-7 - deep learning for brain tumor detection](https://arxiv.org/pdf/2307.16412) - another paper using YOLO (custom architecture)

https://github.com/chetan0220/Brain-Tumor-Detection-using-YOLOv8/blob/main/brain_tumor_object_detection.ipynb - a repository using yolov8 for brain tumor detection

https://github.com/ultralytics/ultralytics- ultralytics documentation

https://stackoverflow.com/questions/75097042/why-is-the-module-ultralytics-not-found-even-after-pip-installing-it-in-the-py

https://forums.fast.ai/t/mixup-data-augmentation/22764

https://stackoverflow.com/questions/56081324/why-are-google-colab-shell-commands-not-working

Claude 3.5 Sonnet - for base and boilerplate code generation

## Setup:
* IDE: Cursor, with Claude 3.5 Sonnet
* Macbook M1 Pro, 32GB RAM, No GPU
* Google Colab, T4 GPU
