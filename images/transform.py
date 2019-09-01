
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

def get_image_transformer(DA_config=DA_default_config, convert_to_tensor=True, p=1.0):
    transformer = Compose([ DA_name2func[k](**v) for k,v in sorted(DA_config.items(), key=lambda x:x[0])], p=p)
    if convert_to_tensor:
        transformer = transformer + [ToTensor()]
    def get_image_transformer_(image):
        return transformer(image=image)['image']
    return get_image_transformer_

def get_filter_transformer(DA_config=DA_default_config, convert_to_tensor=True, p=1.0):
    transformer = Compose([ DA_name2func[k](**v) for k,v in sorted(DA_config.items(), key=lambda x:x[0])], p=p)
    if convert_to_tensor:
        transformer = transformer + [ToTensor()]
    def get_filter_transformer_(image, mask):
        aug = transformer(image=image, mask=mask)
        return aug['image'], aug['mask']
    return get_filter_transformer_

"""
def albumentation_wrapper(image_transformer):
    def augmentator(image, mask):
        aug = image_transformer(image=image, mask=mask)
        return aug['image'], aug['mask']
    return augmentator
"""
