from templates_cls import *
from experiment_classifier import *

if __name__ == '__main__':
    # this requires only a single GPU.
    gpus = [3]
    conf = padchest256_autoenc_cls()
    train_cls(conf, gpus=gpus)

    # after this you can do the manipulation!
