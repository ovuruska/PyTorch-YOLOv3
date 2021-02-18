import os
import xml.etree.ElementTree as ET
import cv2


class Yolov1TxtWriter:

    """

        Creates file with given format :

        path/to/image xmin,ymin,xmax,ymax,class_label



    """


    def __init__(self,
                 root_dir : str,
                 out_path:str
                 ):

        self.class_dict = {}
        self.root_dir = root_dir
        self.label_dir = os.path.join(root_dir,"labels")
        self.image_dir = os.path.join(root_dir,"frames")
        self.out_path = out_path

        with open(os.path.join(self.label_dir,"classes.txt")) as fp:
            for ind, line in enumerate(fp.readlines()):
                class_name = line.strip("\n").lower()
                self.class_dict[class_name] = ind

    def write(self):

        # Image and corresponding label file have a relationship.
        # Let's say img_1 is basename of an image.
        # If corresponding labeling exists, it is found at path,
        #   root_directory/labels/img_1.xml or img_1.txt depending on
        #   YOLO labeling or PascalVOC labeling.

        # At some cases, some images has no objects labeled.
        # For getting them involved in training, path iterations
        # will be done over the images we have.

        with open(self.out_path,"w") as fp:

            images = [
                os.path.join(self.image_dir,image)
                for image in os.listdir(self.image_dir)
                if image.endswith(".png") or image.endswith(".jpg")
            ]

            labels = [
                os.path.join(self.label_dir,label)
                for label in os.listdir(self.label_dir)
            ]


            for img_path in images:
                img_basename = os.path.basename(img_path)
                basename = img_basename.split(".")[0]

                xml_path = os.path.join(self.label_dir,basename+".xml")
                txt_path = os.path.join(self.label_dir,basename+".txt")


                if xml_path in labels:
                    fp.write(self.xml_procedure(
                        xml_path,
                        img_path
                    ))

                elif txt_path in labels:
                    fp.write(self.txt_procedure(
                        txt_path,
                        img_path
                    ))

                else:
                    fp.write(self.no_label(
                        img_path
                    ))

    def no_label(self,image_path):
        return f"{image_path}\n"


    def xml_procedure(self,label_path,image_path):

        ret_line = f"{image_path}"

        etree = ET.parse(label_path)
        root = etree.getroot()


        for obj in root.iter("object"):


            bbox = list(obj.iter("bndbox"))[0]
            xmin = float(list(bbox.iter("xmin"))[0].text)
            xmax = float(list(bbox.iter("xmax"))[0].text)
            ymin = float(list(bbox.iter("ymin"))[0].text)
            ymax = float(list(bbox.iter("ymax"))[0].text)
            class_name = list(obj.iter("name"))[0].text
            class_index = self.class_dict[class_name.lower()]

            x_center = (xmin + xmax) // 2
            y_center = (ymin + ymax) // 2

            width = (xmax - xmin)
            height = (ymax - ymin)

            ret_line += f" {int(x_center)}," \
                        f"{int(y_center)}," \
                        f"{int(width)}," \
                        f"{int(height)}," \
                        f"{class_index}"


        return ret_line + "\n"

    def txt_procedure(self,label_path,image_path):

        ret_line = f"{image_path}"

        img_data = cv2.imread(image_path)
        image_width = img_data.shape[1]
        image_height = img_data.shape[0]


        with open(label_path) as fp:
            for line in fp.readlines():
                class_index, x_center, y_center, width, height = line.split()

                x_center = float(x_center)*image_width
                y_center = float(y_center)*image_height
                width = float(width)*image_width
                height = float(height)*image_height

                ret_line += f" {int(x_center)}," \
                        f"{int(y_center)}," \
                        f"{int(width)}," \
                        f"{int(height)}," \
                        f"{class_index}"

        return ret_line + "\n"