import os
from functools import lru_cache
import keras

MODEL_PATH = "mobicrowd/models/UNIQUE/decoder.h5"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Conv2DTransposeCompat(keras.layers.Conv2DTranspose):
    # Accept `groups` from old configs but ignore it (safe if groups==1)
    def __init__(self, *args, groups=1, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

@lru_cache(maxsize=1)
def get_decoder():
    return keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"Conv2DTranspose": Conv2DTransposeCompat},
    )
