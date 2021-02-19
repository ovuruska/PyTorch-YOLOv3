import imgaug.augmenters as iaa
from .transforms import *

class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.05, 0.05]),
            iaa.Affine( translate_percent=(-0.2,0.2)),  # rotate by -45 to 45 degrees (affects segmaps)
            iaa.Fliplr(0.5),
            iaa.blur_lib.MotionBlur(),
        ], random_order=True)


AUGMENTATION_TRANSFORMS = transforms.Compose([
        AbsoluteLabels(),
        DefaultAug(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ])
