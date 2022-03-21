from tensorflow import keras
from tensorflow_addons.optimizers import extend_with_decoupled_weight_decay


class optimizers:
    adam = keras.optimizers.Adam
    adagrad = keras.optimizers.Adagrad
    sgd_mm = keras.optimizers.SGD
    SGD_MM = 'sgd_mm'


def get_optimizer(opt, lr, wd):
    if opt == optimizers.SGD_MM:
        optimizer = extend_with_decoupled_weight_decay(
            getattr(optimizers, opt))(learning_rate=lr, weight_decay=wd, momentum=0.9)
    else:
        optimizer = extend_with_decoupled_weight_decay(
            getattr(optimizers, opt))(learning_rate=lr, weight_decay=wd)
    return optimizer
