from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate

from keras.layers import Input, \
    UpSampling2D, core, Cropping2D, Activation, SpatialDropout2D
from keras.optimizers import SGD
# from keras.regularizers import l2

from keras.callbacks import Callback, TensorBoard
from keras.initializers import Constant

import keras.backend as K
import numpy as np

init_lr = 0.05
init_mm = 0.2

lr_schedules = [0.02, 0.002, 0.0002]
mm_schedules = [0.9, 0.99, 0.99]
ep_schedules = [10, 14, 18]
mm_p = 0
lr_p = 0

event_path = '/home/bia/retina_vessels/models/GC-22-C/logs/'


import sys
from os.path import dirname
sys.path.append(dirname(__file__))


from layers import adapt2D_crop_layer, get_data_shape, \
    get_crop_from_output, crop2D_layer


def lr_schedule(epoch, actual_lr):
    global lr_p, lr_schedules, ep_schedules

    if epoch > 0 and lr_p < len(ep_schedules) and epoch == ep_schedules[lr_p]:
        new_lr = lr_schedules[lr_p]
        lr_p += 1
        return new_lr
    else:
        return actual_lr


def momentum_schedule(epoch, actual_momentum):
    global mm_p, mm_schedules, ep_schedules

    if epoch > 0 and mm_p < len(ep_schedules) and epoch == ep_schedules[mm_p]:
        new_momentum = mm_schedules[mm_p]
        mm_p += 1
        return new_momentum
    else:
        return actual_momentum


class LearningRateScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """

    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.schedule(epoch, float(K.get_value(self.model.optimizer.lr)))
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        print('Setting lr to', lr, 'on Epoch: ', epoch)
        K.set_value(self.model.optimizer.lr, lr)


class MomentumScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """

    def __init__(self, schedule):
        super(MomentumScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'momentum'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        momentum = \
            self.schedule(epoch,
                          float(K.get_value(self.model.optimizer.momentum)))
        if not isinstance(momentum, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        print('Setting momentum to', momentum, 'on Epoch: ', epoch)
        K.set_value(self.model.optimizer.momentum, momentum)


def callbacks():
    return [LearningRateScheduler(lr_schedule),
            MomentumScheduler(momentum_schedule),
            TensorBoard(log_dir=event_path)]


def create_layer(inp, n_filters, w_regularizer, p=0.0):
    net = Conv2D(n_filters, 3, kernel_regularizer=w_regularizer,
                 padding='valid', kernel_initializer='he_normal',
                 bias_initializer=Constant(value=0.0))(inp)
    if p > 0.0:
        net = SpatialDropout2D(p)(net)
    return Activation('relu')(net)


def create_block(inp, n_filters, n_layers, w_regularizer, p=0.0, p_last=False):
    net = inp
    for _ in range(n_layers - 1):
        net = create_layer(net, n_filters, w_regularizer, p)
    p = p if p_last else 0.0
    net = create_layer(net, n_filters, w_regularizer, p)

    return net


def get_nnet(n_ch, patch, inner_patch):
    patch_height, patch_width = patch
    inner_height, inner_width = inner_patch
    inputs = Input((patch_height, patch_width, n_ch))

    # -- We add 3 given the number of convolutions that follows.
    h = (patch_height - inner_height) // 2 - 3
    w = (patch_width - inner_width) // 2 - 3

    W_regularizer = None
    p = 0.2
    p_pred = 0.15

    n_layers_block = 3

    # -- Block #1
    net1 = create_block(inputs, 32, n_layers_block, W_regularizer, p)
    pool1 = MaxPooling2D(pool_size=(2, 2))(net1)

    # -- Block #2
    net2 = create_block(pool1, 64, n_layers_block, W_regularizer, p)
    pool2 = MaxPooling2D(pool_size=(2, 2))(net2)

    # -- Block #3
    net3 = create_block(pool2, 128, n_layers_block, W_regularizer, p)

    up = UpSampling2D(size=(2, 2))(net3)
    net2 = adapt2D_crop_layer(bigger_inp=net2, smaller_inp=up)
    up1 = Concatenate(axis=3)([up, net2])

    # -- Block #4
    net4 = create_block(up1, 64, n_layers_block, W_regularizer, p)

    up = UpSampling2D(size=(2, 2))(net4)
    net1 = adapt2D_crop_layer(bigger_inp=net1, smaller_inp=up)
    up2 = Concatenate(axis=3)([up, net1])

    cropping = get_crop_from_output(
        inp=up2, output_shape=[inner_height + (2 * n_layers_block),
                               inner_width + (2 * n_layers_block)],
        patch_topology='2D')
    crop = crop2D_layer(up2, cropping)

    # -- Block #5
    net5 = create_block(crop, 32, n_layers_block, W_regularizer, p_pred)

    conv6 = Conv2D(2, 1, kernel_regularizer=W_regularizer,
                   padding='valid', kernel_initializer='he_normal',
                   bias_initializer=Constant(value=0.0))(net5)
    conv6 = core.Reshape((inner_height * inner_width, 2))(conv6)
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    sgd = SGD(lr=init_lr, decay=1e-6, momentum=init_mm, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
