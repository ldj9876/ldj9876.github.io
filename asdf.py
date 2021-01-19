import keras
import numpy as np
import tensorflow as tf

class args:
    num_class = 10
    num_batch = 40
    batch_size = 125
    num_valid = 1000

    n_craftstep = 60
    craft_rate = 200
    craft_rate_drop_period = 20
def get_resnet(model_name,input_shape, num_class=10):
    if model_name == 'resnet50':
        model = keras.applications.ResNet50(include_top=True, weights=None,classes=num_class,input_shape=input_shape)
    elif model_name == 'resnet101':
        model = keras.applications.ResNet50(include_top=True, weights=None,classes=num_class,input_shape=input_shape)
    elif model_name == 'resnet152':
        model = keras.applications.ResNet50(include_top=True, weights=None,classes=num_class,input_shape=input_shape)

    else:
        raise ValueError(f'Invalid ResNet model chosen: {model_name}.')

    return model

loss = tf.reduce_mean


n_craftstep = args.n_craftstep
craft_rate = args.craft_rate
craft_rate_drop_period = args.craft_rate_drop_period

for craftstep in range(n_craftstep):
    if craftstep % craft_rate_drop_period == 0:
        craft_rate = craft_rate / 10

