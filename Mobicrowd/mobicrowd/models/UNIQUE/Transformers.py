from torchvision import transforms # type: ignore
from PIL import Image # type: ignore

class AdaptiveResize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        h, w = img.size
        if h < self.size or w < self.size:
            return img
        else:
            return transforms.Resize(self.size, self.interpolation)(img)