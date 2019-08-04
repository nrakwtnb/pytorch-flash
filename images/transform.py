
from albumentations import ShiftScaleRotate, HorizontalFlip, VerticalFlip
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensor

DA_name2func = {
    "H-flip" : HorizontalFlip,
    "V-flip" : VerticalFlip,
    "Affine" : ShiftScaleRotate,
}

DA_default_config = {
    "H-flip" : { "p" : 0.5},
    "V-flip" : { "p" : 0.5},
    "Affine" : { "shift_limit" : 0.0625, "scale_limit" : 0.1, "rotate_limit" : 45 },
}

def get_image_transformer(DA_config=DA_default_config, p=1.0):
    transformer = Compose([ DA_name2func[k](**v) for k,v in sorted(DA_config.items(), key=lambda x:x[0])]+[ToTensor()], p=p)
    def get_image_transformer_(image):
        return transformer(image=image)['image']
    return get_image_transformer_
