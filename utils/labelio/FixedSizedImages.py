import cv2
import os
import numpy as np


class FixedSizedImages():


    @staticmethod
    def read_text_file(text_path : str,filter_classes = None):

        filenames = []
        annotations = []

        with open(text_path) as fp:

            for line in fp.readlines():

                img_annotations = []

                words = line.strip("\n").split()
                filenames.append(words[0])

                for word in words[1:]:
                    xmin,ymin,xmax,ymax,cls = word.split(",")

                    xmin = int(xmin)
                    xmax = int(xmax)
                    ymin = int(ymin)
                    ymax = int(ymax)
                    cls = int(cls)

                    if filter_classes is not None and (str(cls) not in filter_classes and cls not in filter_classes):
                        continue
                    else:
                        img_annotations.append((xmin, ymin, xmax, ymax, cls))


                annotations.append(img_annotations)

        return filenames,annotations

    def __init__(self,txt,width = 512,height = 384,filter_classes = None):

        self.filenames,self.annotations = self.read_text_file(txt,filter_classes)
        self.height = height
        self.width = width


    def get_scale_ratio(self,img):
        height_ratio = img.shape[0] / self.height
        width_ratio = img.shape[1] / self.width

        max_ratio = max(height_ratio, width_ratio)
        return max_ratio





    def scale_img(self,img,img_annotations):

        ratio = self.get_scale_ratio(img)

        new_img = np.zeros((self.height,self.width,3)).astype(np.uint8)
        img = cv2.resize(img, (int(img.shape[1] // ratio), int(img.shape[0] // ratio)))

        x_offset = (self.width - img.shape[1])
        y_offset = (self.height - img.shape[0])

        left_offset = x_offset // 2 + x_offset % 2
        top_offset = y_offset // 2 + y_offset % 2

        new_img[top_offset:top_offset+img.shape[0],left_offset:left_offset+img.shape[1],:] = img

        ret_annotations = [
            [
                anno[4],
                int((anno[0]/ratio)+left_offset)/self.width,
                int((anno[1]/ratio)+top_offset)/self.height,
                int((anno[2]/ratio))/self.width,
                int((anno[3]/ratio))/self.height,
            ]
            for anno in img_annotations
        ]

        return new_img,ret_annotations

    def write_dataset(self,images_root,labels_root,image_list_path):


        with open(image_list_path,"w") as image_list_fp:
            for filepath,img_annotations in zip(self.filenames,self.annotations):
                img = cv2.imread(filepath)
                img,img_annotations = self.scale_img(img,img_annotations)

                image_write_path = os.path.join(images_root,os.path.basename(filepath))
                cv2.imwrite(image_write_path,img)

                image_list_fp.write(f"{image_write_path}\n")


                img_basename = os.path.basename(filepath)
                img_name = img_basename.split(".")[0]
                with open(os.path.join(labels_root,img_name+".txt"),"w") as label_fp:

                    for anno in img_annotations:
                        label_fp.write(f"{anno[0]} {anno[1]} {anno[2]} {anno[3]} {anno[4]}\n")





