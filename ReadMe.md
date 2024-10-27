
Links:
https://www.mdpi.com/2076-3417/13/16/9158 - yolov8 for brain tumor detection

[https://www.nature.com/articles/s41598-024-57970-7 - deep learning for brain tumor detection](https://arxiv.org/pdf/2307.16412) - another paper using yolo (custom architecture)



we chose to proceed with yolov8



Dataset:

It looks like there are three categories/classes:

1. "Tumor" (with id: 0)
2. "0" (with id: 1, supercategory: "Tumor")
3. "1" (with id: 2, supercategory: "Tumor")

For now we will treat these as three separate classes due to interest of time as later we can easily convert back to one class if necessary in the application layer but it is better to understand what exactly these are.

We visualize the dateset and verify that the data and bounding boxes seem fine. (It is however noteworthy that the bounding boxes are not very tight and there is a lot of space between the tumor and the edge of the bounding box, so if given more time we might want to take this into account)

We also verify the conversion to yolo format. 

All looks good! Time to train a basic model (no augmentation or image preprocessing).

We use a pretrained model (on COCO dataset) and train it on our dataset