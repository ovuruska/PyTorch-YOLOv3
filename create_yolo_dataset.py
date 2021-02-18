from utils.labelio.FixedSizedImages import FixedSizedImages
from utils.labelio.YoloV1TxtWriter import Yolov1TxtWriter
from utils.labelio.YoloV4TXTWriter import YoloV4TXTWriter
import os

if __name__ == "__main__":


    # train_root = os.path.join("data","fire_detection", "train")
    # val_root = os.path.join("data","fire_detection", "val")
    #
    # train_writer = Yolov1TxtWriter(train_root,os.path.join("train_yolo.txt"))
    # train_writer.write()
    # val_writer = Yolov1TxtWriter(val_root,os.path.join("val_yolo.txt"))
    # val_writer.write()


    images_root = os.path.join("data","fire_data", "images")
    labels_root = os.path.join("data","fire_data", "labels")
    train_images_txt = os.path.join("data","fire_data","train.txt")
    val_images_txt = os.path.join("data","fire_data","valid.txt")




    FixedSizedImages("train_yolo.txt",filter_classes=[0])\
        .write_dataset(
            images_root,
            labels_root,
            train_images_txt
    )

    FixedSizedImages("val_yolo.txt", filter_classes=[0]) \
        .write_dataset(
        images_root,
        labels_root,
        val_images_txt
    )

