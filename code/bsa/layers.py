import numpy as np

from keras.layers import Input
from keras.layers.convolutional import Conv3D, Conv2D
from keras.layers import UpSampling3D, UpSampling2D, Cropping3D, \
    Cropping2D, Activation, SpatialDropout3D, SpatialDropout2D, Reshape, \
    BatchNormalization, MaxPooling3D, MaxPooling2D, ConvLSTM2D, Permute, \
    Reshape
from keras.layers.merge import add
from keras.layers.wrappers import TimeDistributed

from keras.backend import image_data_format, int_shape


def get_data_shape(layer):
    return tuple(int_shape(layer)[1:])


def input_layer(shape, batch_layout='channels_last'):
    assert type(shape) is tuple
    assert type(batch_layout) is str

    inputs = Input(shape)

    return inputs


def get_effective_shape(input_shape):
    data_format = image_data_format()
    if data_format == 'channels_last':
        return input_shape[0: -1], input_shape[-1]
    else:
        return input_shape[1:], input_shape[1]


def append_channels(shape, n_channels):
    data_format = image_data_format()
    if data_format == 'channels_last':
        return tuple(shape) + tuple([n_channels])
    else:
        return tuple([n_channels]) + tuple(shape)


def conv_output_shape(input_shape, kernel_size, strides, dilation_rate,
                      padding, filters):
    # For now it does not account for strides and dilation rate
    if isinstance(strides, int):
      assert strides == 1
    else:
      for s in strides:
        assert s == 1

    if isinstance(dilation_rate, int):
      assert dilation_rate == 1
    else:
      for d in dilation_rate:
        assert d == 1

    input_shape, _ = get_effective_shape(input_shape=input_shape)

    if type(kernel_size) is int:
        kernel_size = [kernel_size] * len(input_shape)

    assert len(input_shape) == len(kernel_size)

    # currently only works with odd kernel_size
    if padding == 'valid':
        for s in kernel_size:
            assert s % 2 != 0

        conv_out_shape = [i_s - (k_s // 2) * 2 for i_s, k_s in
                          zip(input_shape, kernel_size)]

    elif padding == 'same':
        conv_out_shape = input_shape

    else:
        raise ValueError('Unrecognized padding for convolution', padding)

    return append_channels(shape=conv_out_shape, n_channels=filters)

#conv3D_layer(inp, **pars)


def conv3D_layer(inp, filters, kernel_size, strides=1, padding='valid',
                 dilation_rate=1, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None):

    assert isinstance(filters, int)
    assert isinstance(kernel_size, (int, list, tuple))
    assert type(strides) in (int, list, tuple)
    assert type(dilation_rate) in (int, list, tuple)

    # TODO: generalize for different dilation and strides
    assert dilation_rate == 1
    assert strides == 1

    input_shape = get_data_shape(inp)

    conv_op = Conv3D(filters=filters, kernel_size=kernel_size,
                     strides=strides, padding=padding,
                     dilation_rate=dilation_rate, activation=activation,
                     use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     activity_regularizer=activity_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint)(inp)

    output_shape = conv_output_shape(input_shape=input_shape,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     dilation_rate=dilation_rate,
                                     padding=padding,
                                     filters=filters)

    return conv_op


def conv2D_layer(inp, filters, kernel_size, strides=1, padding='valid',
                 dilation_rate=1, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None):

    assert type(filters) is int
    assert type(kernel_size) in (int, list, tuple)
    assert type(strides) in (int, list, tuple)
    assert type(dilation_rate) in (int, list, tuple)

    # TODO: generalize for different dilation and strides
    assert dilation_rate == 1
    assert strides == 1

    input_shape = get_data_shape(inp)

    conv_op = Conv2D(filters=filters, kernel_size=kernel_size,
                     strides=strides, padding=padding,
                     dilation_rate=dilation_rate, activation=activation,
                     use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     activity_regularizer=activity_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint)(inp)

    output_shape = conv_output_shape(input_shape=input_shape,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     dilation_rate=dilation_rate,
                                     padding=padding,
                                     filters=filters)

    return conv_op


def upsample_output_shape(input_shape, size):

    input_shape, n_channels = get_effective_shape(input_shape=input_shape)

    if type(size) is int:
        size = [size] * len(input_shape)

    assert len(input_shape) == len(size)

    up_out_shape = [i_s * s for i_s, s in zip(input_shape, size)]

    return append_channels(shape=up_out_shape, n_channels=n_channels)


def upsample3D_layer(inp, size=(2, 2, 2), data_format=None):

    assert type(size) in (int, list, tuple)
    assert len(size) == 3

    input_shape = get_data_shape(inp)

    up_op = UpSampling3D(size=size, data_format=data_format)(inp)

    output_shape = upsample_output_shape(input_shape=input_shape, size=size)

    return up_op


def upsample2D_layer(inp, size=(2, 2), data_format=None):

    assert type(size) in (int, list, tuple)
    assert len(size) == 2

    input_shape = get_data_shape(inp)

    up_op = UpSampling2D(size=size, data_format=data_format)(inp)

    output_shape = upsample_output_shape(input_shape=input_shape, size=size)

    return up_op


def get_cropping_tuple(cropping, patch_topology):
    assert patch_topology in ('2D', '3D')

    dim = 2 if patch_topology == '2D' else 3

    if type(cropping) is int:
        return tuple((cropping, cropping) for _ in range(dim))

    if type(cropping[0]) is int:
        return tuple((crop, crop) for crop in cropping)

    if len(cropping[0]) == 2:
        return cropping

    raise ValueError('Unrecognized cropping values', cropping)


def cropping_output_shape(input_shape, cropping):
    input_shape, n_channels = get_effective_shape(input_shape=input_shape)

    output_shape = [i_s - crop[0] - crop[1] for i_s, crop in
                    zip(input_shape, cropping)]

    return append_channels(shape=output_shape, n_channels=n_channels)


def get_crop_from_output(inp, output_shape, patch_topology):
    input_shape = get_data_shape(inp)
    input_shape, _ = get_effective_shape(input_shape=input_shape)

    assert len(input_shape) == len(output_shape)
    total_crop = [(i_s - o_s) // 2 for i_s,
                  o_s in zip(input_shape, output_shape)]

    return get_cropping_tuple(cropping=total_crop,
                              patch_topology=patch_topology)


def crop3D_layer(inp, cropping=((1, 1), (1, 1), (1, 1)), data_format=None,
                 time_dist=False):
    assert type(cropping) in (int, tuple)

    input_shape = get_data_shape(inp)

    cropping = get_cropping_tuple(cropping=cropping, patch_topology='2D')

    if time_dist:
        crop_op = TimeDistributed(Cropping3D(cropping=cropping))(inp) 
    else:
        crop_op = Cropping3D(cropping=cropping)(inp) 

    return crop_op


def crop2D_layer(inp, cropping=((1, 1), (1, 1)), data_format=None,
                 time_dist=False):
    assert type(cropping) in (int, tuple)

    input_shape = get_data_shape(inp)

    cropping = get_cropping_tuple(cropping=cropping, patch_topology='2D')

    if time_dist:
        crop_op = TimeDistributed(Cropping2D(cropping=cropping))(inp) 
    else:
        crop_op = Cropping2D(cropping=cropping)(inp) 

    return crop_op


def compute_cropping(bigger_inp, smaller_inp, time_dist=False):
    # it adapts the output to be in the valid shape as smaller_inp
    big_shape = get_data_shape(bigger_inp)
    big_shape = big_shape[1:] if time_dist else big_shape
    big_shape, _ = get_effective_shape(input_shape=big_shape)
    small_shape = get_data_shape(smaller_inp)
    small_shape = small_shape[1:] if time_dist else small_shape
    small_shape, _ = get_effective_shape(input_shape=small_shape)

    cropping = []

    for i in range(0, len(big_shape)):
        assert big_shape[i] >= small_shape[i]

        side_crop = big_shape[i] - small_shape[i]

        cropping.append((side_crop // 2, side_crop // 2))

    cropping = tuple(cropping)

    return cropping


def adapt3D_crop_layer(bigger_inp, smaller_inp, data_format=None,
                       time_dist=False):
    cropping = compute_cropping(bigger_inp, smaller_inp, time_dist=time_dist)

    crop_op = crop3D_layer(bigger_inp, cropping=cropping,
                           data_format=data_format, time_dist=time_dist)

    return crop_op


def adapt2D_crop_layer(bigger_inp, smaller_inp, data_format=None,
                       time_dist=False):
    cropping = compute_cropping(bigger_inp, smaller_inp, time_dist=time_dist)

    crop_op = crop2D_layer(bigger_inp, cropping=cropping,
                           data_format=data_format, time_dist=time_dist)

    return crop_op


def activation_layer(inp, activation):
    act_op = Activation(activation)(inp)
    output_shape = get_data_shape(inp)

    return act_op


def spatial_dropout3D_layer(inp, rate, data_format=None):
    drop_op = SpatialDropout3D(rate=rate, data_format=data_format)(inp)
    output_shape = get_data_shape(inp)

    return drop_op


def spatial_dropout2D_layer(inp, rate, data_format=None):
    drop_op = SpatialDropout2D(rate=rate, data_format=data_format)(inp)
    output_shape = get_data_shape(inp)

    return drop_op


def reshape_layer(inp, target_shape):
    assert type(target_shape) is tuple

    reshape_op = Reshape(target_shape=target_shape)(inp)

    return reshape_op


def batch_normalization_layer(inp, axis=-1, momentum=0.99, epsilon=0.001,
                              center=True, scale=True,
                              beta_initializer='zeros',
                              gamma_initializer='ones',
                              moving_mean_initializer='zeros',
                              moving_variance_initializer='ones',
                              beta_regularizer=None, gamma_regularizer=None,
                              beta_constraint=None, gamma_constraint=None):

    norm_op = BatchNormalization(
        axis=axis, momentum=momentum, epsilon=epsilon, center=center,
        scale=scale, beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint)(inp)

    return norm_op


def add_layer(inps):
    assert type(inps) in (list, tuple)
    assert len(inps) > 1

    add_inputs = [inp for inp in inps]
    add_op = add(inputs=add_inputs)

    return add_op


def pool_output_shape(inp, pool_size, strides, padding):
    # For now it does not account for strides and padding valid
    assert strides == 1 or strides == None
    assert padding == 'valid'

    input_shape = get_data_shape(inp)

    input_shape, n_channels = get_effective_shape(input_shape=input_shape)

    pool_output_shape = [i_s // p_s for i_s, p_s in zip(input_shape,
                                                        pool_size)]

    return append_channels(shape=pool_output_shape, n_channels=n_channels)


def max_pooling3D_layer(inp, pool_size=(2, 2, 2), strides=None,
                        padding='valid', data_format=None):
    assert type(pool_size) is tuple
    assert len(pool_size) == 3
    if strides != None:
        assert type(strides) is tuple
        assert len(strides) == 3

    pool_op = MaxPooling3D(pool_size=pool_size, strides=strides,
                           padding=padding, data_format=data_format)(inp)

    output_shape = pool_output_shape(inp=inp, pool_size=pool_size,
                                     strides=strides, padding=padding)

    return pool_op


def max_pooling2D_layer(inp, pool_size=(2, 2), strides=None,
                        padding='valid', data_format=None):
    assert type(pool_size) is tuple
    assert len(pool_size) == 2
    if strides != None:
        assert type(strides) is tuple
        assert len(strides) == 2

    pool_op = MaxPooling2D(pool_size=pool_size, strides=strides,
                           padding=padding, data_format=data_format)(inp)

    output_shape = pool_output_shape(inp=inp, pool_size=pool_size,
                                     strides=strides, padding=padding)

    return pool_op


def conv_LSTM2D_layer(inp, filters, kernel_size, strides=(1, 1),
                      padding='valid', dilation_rate=(1, 1),
                      activation='tanh', recurrent_activation='hard_sigmoid',
                      use_bias=True, kernel_initializer='glorot_uniform',
                      recurrent_initializer='orthogonal',
                      bias_initializer='zeros', unit_forget_bias=True,
                      kernel_regularizer=None, recurrent_regularizer=None,
                      bias_regularizer=None, activity_regularizer=None,
                      kernel_constraint=None, recurrent_constraint=None,
                      bias_constraint=None, return_sequences=False,
                      go_backwards=False, stateful=False, dropout=0.):

    assert isinstance(filters, int)
    assert isinstance(kernel_size, (int, list, tuple))
    assert isinstance(strides, (int, list, tuple))
    assert isinstance(dilation_rate, (int, list, tuple))
    assert padding in ('valid', 'same')

    data_format = image_data_format()

    conv_lstm_op = ConvLSTM2D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding=padding, dilation_rate=dilation_rate,
        activation=activation, recurrent_activation=recurrent_activation,
        use_bias=use_bias, kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        return_sequences=return_sequences, go_backwards=go_backwards,
        stateful=stateful, dropout=dropout)(inp)

    input_shape = get_data_shape(inp)

    # if data_format == 'channels_last':
    #     conv_inp_shape = input_shape[1:3]
    # else:
    #     conv_inp_shape = input_shape[2:]

    time_dim = input_shape[0]
    conv_input_shape = input_shape[1:]

    output_shape = conv_output_shape(input_shape=conv_input_shape,
                                     kernel_size=kernel_size, strides=strides,
                                     dilation_rate=dilation_rate,
                                     padding=padding, filters=filters)

    if return_sequences:
        output_shape = tuple([time_dim]) + tuple(output_shape)

    else:
        output_shape = tuple([1]) + tuple(output_shape)

        conv_lstm_op = Reshape(target_shape=output_shape)(conv_lstm_op)

    return conv_lstm_op


def time_distributed_layer(inp, layer):
    data_format = image_data_format()

    conv_inp_shape = inp[1][2:]
    input_shape = inp[1][1:]

    time_distributed_op = TimeDistributed(
        layer, input_shape=input_shape)((inp[0], conv_inp_shape))

    output_shape = tuple([inp[1][1]]) + tuple(time_distributed_op[1])

    return time_distributed_op


def permute_layer(inp, dims):
    input_shape = get_data_shape(inp)
    
    assert len(dims) == len(input_shape)
    assert np.all(dims) > 0  # indexing starts in 1, here

    output_shape = tuple([input_shape[dim - 1] for dim in dims])

    permute_op = Permute(dims=dims, input_shape=input_shape)(inp)

    return permute_op
