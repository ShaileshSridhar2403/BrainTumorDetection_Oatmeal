
Process and Notes:

First I check out the images and think about the task. This is object detection for brain tumors. We have an adequate number of training images (~3000) so nothing to worry about there. I do a brief literature survey to see what rough approach is known to work well.

Links:
https://www.mdpi.com/2076-3417/13/16/9158 - yolov8 for brain tumor detection

[https://www.nature.com/articles/s41598-024-57970-7 - deep learning for brain tumor detection](https://arxiv.org/pdf/2307.16412) - another paper using yolo (custom architecture)

It seems that yolo architectures are known to perform well on this task. yolov8 is easy to set up, has a "nano" sized model which can be trained fast and we decide to use this for our task.



Dataset:

It looks like there are three categories/classes:

1. "Tumor" (with id: 0)
2. "0" (with id: 1, supercategory: "Tumor")
3. "1" (with id: 2, supercategory: "Tumor")

For now we will treat these as three separate classes due to interest of time as later we can easily convert back to one class if necessary in the application layer but it is better to understand what exactly these are.

We visualize the dateset and verify that the data and bounding boxes seem fine. (It is however noteworthy that the bounding boxes are not very tight and there is a lot of space between the tumor and the edge of the bounding box, so if given more time we might want to take this into account)

We also verify the conversion to yolo format. 

All looks good! Time to train a basic model

We use a pretrained model (on COCO dataset) and train it on our dataset. We use basic augmentations during training, defaults for the ultralytics library, such as blurring, median blurring, and CLAHE.



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

We notice that both augmentation seem to help.

We also notice that the loss curve is still decreasing. Hence it is a good idea to train for more epochs.

In our final run we increase the number of epochs to 15, use flip and mixup augmentations, scale the image size back to 640x640 and run. Ideally we would like to run for more epochs, until the curve plateaus, but due to time constraints we stop at 15.

                 Class     Images  Instances      P          R          mAP50  mAP50-95
                   all        301        302      0.811      0.758      0.826      0.493
                     0        163        163      0.776      0.595       0.72      0.393
                     1        138        139      0.846      0.921      0.933      0.594

We get reasonable resuls, with an mAP of 0.933 at a threshold of 0.5 as well as reasonable precision and recall.

We still see a couple of false positives and negatives, some of which will go away with more training.

I also tried to use TIDE toolbox(https://dbolya.github.io/tide/paper.pdf) to look a little deeper at the types of errors our model makes but was unable to debug it and get it working within time constraints. The code is attached.

Taking these further steps, as well as addressingthe slight disparity between the two classes, we can see that we can further improve the model
    


References and Resources Used:

https://www.mdpi.com/2076-3417/13/16/9158 - yolov8 for brain tumor detection

[https://www.nature.com/articles/s41598-024-57970-7 - deep learning for brain tumor detection](https://arxiv.org/pdf/2307.16412) - another paper using yolo (custom architecture)

https://github.com/chetan0220/Brain-Tumor-Detection-using-YOLOv8/blob/main/brain_tumor_object_detection.ipynb - a repository using yolov8 for brain tumor detection

https://github.com/ultralytics/ultralytics- ultralytics documentation

https://stackoverflow.com/questions/75097042/why-is-the-module-ultralytics-not-found-even-after-pip-installing-it-in-the-py

https://forums.fast.ai/t/mixup-data-augmentation/22764

https://stackoverflow.com/questions/56081324/why-are-google-colab-shell-commands-not-working

Claude 3.5 Sonnet - for base and boilerplate code generation

Setup:
IDE: Cursor, with Claude 3.5 Sonnet
Macbook M1 Pro, 32GB RAM, No GPU
Google Colab, T4 GPU