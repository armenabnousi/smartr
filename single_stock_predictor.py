#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
#import mdn
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import h5py


# In[203]:


'''
stock_name = "MSFT"
weekly = False
window_size = 30
batch_size = 32
shuffle_buffer = 100000

distributional = False
'''


# In[ ]:


def predict_tomorrow(stock_name, d, model_outdir, weekly = False, window_size = 30, batch_size = 32, training_points = None, distributional = False, epochs = 20, sd_estimate_required = True, shuffle_buffer = None, training_verbosity = 2, look_ahead_window = 1):
    d = d.loc[:,[stock_name]]
    if training_points is None:
        training_points = 350 if weekly else 1500
    if weekly:
        d = convert_to_weekly(d)
    if shuffle_buffer is None:
        shuffle_buffer = d.shape[0] * 3
    train_d = d.iloc[:training_points, :]
    valid_d = d.iloc[training_points:, :]
    train_df = windowed_dataset(train_d.to_numpy().reshape((len(train_d,))), window_size, batch_size, shuffle_buffer, look_ahead_window = look_ahead_window)
    valid_df = windowed_dataset(valid_d.to_numpy().reshape((len(valid_d,))), window_size, batch_size, shuffle_buffer, look_ahead_window = look_ahead_window, shuffle = False)
    normalization_factors = train_d.max()
    model = train_model(train_df, valid_df, model_outdir, stock_name, epochs, distributional, window_size, training_verbosity)
    if sd_estimate_required:
        train_sd_estimate, train_forecasts = get_sd_estimate(model, train_d, window_size, look_ahead_window = look_ahead_window)
        valid_sd_estimate, valid_forecasts = get_sd_estimate(model, valid_d, window_size, look_ahead_window = look_ahead_window)
    else:
        train_sd_estimate = None
        train_forecasts = None
        valid_sd_estimate = None
        valid_forecasts = None	

    tomorrows_prediction = model.predict(np.array(d[stock_name])[-window_size:].reshape(1, window_size, 1))[0, 0]
    return tomorrows_prediction, train_sd_estimate, train_forecasts, valid_sd_estimate, valid_forecasts


# In[202]:


# this function is copied from the deeplearning.ai course on time-series analysis on coursera! with minor modifications
def windowed_dataset(series, window_size, batch_size, shuffle_buffer, shuffle = True, look_ahead_window = 1):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + look_ahead_window, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + look_ahead_window))
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (tf.expand_dims(window[:-look_ahead_window], axis = -1), window[-1]))
        #dataset = dataset.map(lambda window: (window[0].reshape((len(window[0], 1))), window[1]))
    else:
        dataset = dataset.map(lambda window: (tf.expand_dims(window[:-look_ahead_window], axis = -1), window[-1]))
        #dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


# In[205]:


def convert_to_weekly(d):
    d.reset_index(drop = False, inplace = True)
    d['weekday'] = d['Date'].dt.day_name()
    d = d[d['weekday'] == "Friday"]
    d.drop(['Date', 'weekday'], axis = 1, inplace = True)
    print(d.shape)
    print(d.head())
    d.reset_index(drop = True, inplace = True)
    return d


# In[13]:


def layer_normalize(x, factors = None, denorm = False, input_mean = None, input_std = None):
    if factors is None:
        factors = tf.reduce_max(x, axis = 0, keepdims = False)
        #tf.print(factors)
        tf.print("pre mod shape of x")
        tf.print(tf.shape(x))
        tf.print("shape of factors")
        tf.print(tf.shape(factors))
    input_mean = tf.math.reduce_mean(x, axis = 0)
    input_std = tf.keras.backend.std(x)
    if denorm:
        #x += 1
        factors = 1. / factors
    x /= factors
    '''
    if not denorm:
        if len(x.get_shape()) == 3:
            #tf.print('mean', input_mean)
            #tf.print('std')
            #tf.print('std', input_std)
            #tf.print('oldx')
            #tf.print('shape',x.get_shape())
            #tf.print(x)
            x = (x - input_mean)/input_std
    else:
        #tf.print('mean')       
        #tf.print(len(x.get_shape()))
        x = x * input_std + input_mean
    '''
    #x = (x - tf.keras.backend.mean(x)) / tf.keras.backend.std(x)
    #tf.print("post mod shape of x")
    #tf.print(tf.shape(x))
    return x


# In[ ]:


def create_model(distributional, window_size):
    tf.compat.v1.reset_default_graph()
    #output_size = train_d.shape[1]
    tfd = tfp.distributions
    output_dense_size = 2 if distributional else 1

    def activations(l, input_mean = None, input_std = None, window_size = None):
        l_0 = (tf.keras.activations.linear(l[...,0])) * input_mean #* input_std) + input_mean #* normalization_factors
        #l_1 = std_multiplier + 
        l_1 = ((tf.keras.activations.linear(tf.abs(l[...,1])))) + 1e-6 # * input_std + input_mean) / window_size) / 0.5
        #tf.print(l_1)
        lnew = tf.stack([l_0, l_1], axis = 1)
        return lnew

    def simple_activations(l):
        l_0 = tf.keras.activations.linear(l[...,0])
        l_1 = tf.keras.activations.elu(l[...,1])
        lnew = tf.stack([l_0, l_1], axis = 1)
        return lnew

    initializers = "glorot_normal"
    activation_name = 'relu'
    model = tf.keras.models.Sequential([
      #tf.keras.layers.LayerNormalization(axis = 0),
      #tf.keras.layers.Lambda(layer_normalize, arguments={'factors': normalization_factors, 'input_mean' : np.mean(train_d)[0], 'input_std': np.std(train_d)[0], 'denorm' : False}, input_shape = (window_size,)),
      #tf.keras.layers.GRU(32, return_sequences = True, kernel_initializer = initializers, activation = activation_name, input_shape = (window_size, 1)), 
      #tf.keras.layers.Conv1D(128, kernel_size = 3),
      #tf.keras.layers.AveragePooling1D(pool_size = 3, padding = 'valid'),
      #tf.keras.layers.LSTM(64, return_sequences=True, kernel_initializer = initializers, activation = activation_name),
      #tf.keras.layers.LSTM(256, return_sequences=True, kernel_initializer = initializers),
      #tf.keras.layers.Dropout(0.5),
      #tf.keras.layers.LSTM(64, return_sequences=True, kernel_initializer = initializers, activation = activation_name),
      #tf.keras.layers.LSTM(128, return_sequences=True, kernel_initializer = initializers),
      #tf.keras.layers.Dropout(0.1),
      #tf.keras.layers.GRU(32, return_sequences=True, kernel_initializer = initializers, activation = activation_name, input_shape = (window_size, 1)),
      tf.keras.layers.SimpleRNN(128, kernel_initializer = initializers, activation = activation_name, input_shape = (window_size, 1), return_sequences = True),
      #tf.keras.layers.Dropout(0.1),
      tf.keras.layers.SimpleRNN(128, kernel_initializer = initializers, activation = activation_name, return_sequences = False),
      #tf.keras.layers.Dropout(0.5),
      #tf.keras.layers.SimpleRNN(128, kernel_initializer = initializers, activation = activation_name),
      #tf.keras.layers.LSTM(32, kernel_initializer = initializers, activation = activation_name),
      #tf.keras.layers.Dense(output_dense_size, activation = tf.keras.layers.Activation(lambda x: activations(x, np.mean(train_d)[0], np.std(train_d)[0], window_size)), kernel_initializer = "he_normal"),
      #tf.keras.layers.Lambda(layer_normalize, arguments={'factors': normalization_factors, 'denorm' : True}),
      #tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=tf.abs(t[..., 0]), scale=0.01*(tf.abs(t[..., 1]))))#-t[...,0]))))#t[...,1])) 
      #                         #scale=(tf.keras.backend.std[...,1])))
      #tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[:,0], scale = t[...,0] + tf.keras.backend.std(t[:,1])))
      #                         #scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
      #tfp.layers.IndependentNormal(output_size)
      #tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))
    ])

    if distributional:
        model.add(tf.keras.layers.Dense(
            output_dense_size, activation = 'linear',
            #activation = tf.keras.layers.Activation(lambda x: activations(x, np.mean(train_d)[0], np.std(train_d)[0], window_size)),
            #activation = tf.keras.layers.Activation(lambda x: simple_activations(x)),
            kernel_initializer = "he_uniform"))
        model.add(tfp.layers.DistributionLambda(lambda t: 
                                                tfd.Normal(
                                                    loc=tf.abs(t[..., 0]), 
                                                    scale= 1e-6 + tf.abs(t[..., 1]) #* normalization_factors
                                                )))
    else: 
        #model.add(tf.keras.layers.Dense(128, activation = 'linear'))
        model.add(tf.keras.layers.Dense(1, activation = 'linear', kernel_initializer = initializers))
        #model.add(tf.keras.layers.Lambda(layer_normalize, arguments={'factors': normalization_factors, 'input_mean' : np.mean(train_d)[0], 'input_std': np.std(train_d)[0], 'denorm' : True}))
    #model.add(tf.keras.layers.Lambda(layer_normalize, arguments={'factors': normalization_factors, 'denorm' : True}))
    return model


# In[89]:


def train_model(train_df, valid_df, model_outdir, stock_name, epochs, distributional, window_size, training_verbosity):
    model = create_model(distributional, window_size)
    negloglik = lambda y, p_y: -p_y.log_prob(y)
    loss_function = negloglik if distributional else 'mse' 
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_function)
    model_output = model_outdir + '/' + stock_name + '.mdl_wts.hdf5'
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_output, save_best_only=True, monitor='val_loss', mode='min')
    model.fit(train_df, epochs=epochs, validation_data = valid_df, callbacks = [model_checkpoint], verbose = training_verbosity)
    model.load_weights(model_output)
    return model


# In[168]:


def get_sd_estimate(model, train_d, window_size, look_ahead_window = 1):
    forecasts_x = []
    sds = []
    #indices = [j for j in range(x.get_shape()[0])]      
    #print(indices)
    vs = np.array(train_d)
    #xs = list(x[i,:,:] for i in indices)
    for time in range(len(train_d) - window_size - look_ahead_window + 1):
        prediction = model.predict(vs[time:time+window_size].reshape((1, window_size, 1)))
        forecasts_x.append(prediction)
        sds.append(prediction - vs[time+window_size+look_ahead_window-1])
    sd_estimate = np.sqrt(np.sum(np.array(sds)**2) / (train_d.shape[0] - window_size -look_ahead_window + 1 - 1))
    #print(train_d.shape[0] - window_size - 1)
    forecasts_x = np.array(forecasts_x)[:,0].reshape((train_d.shape[0] - window_size - look_ahead_window + 1))
    return sd_estimate, forecasts_x


# In[178]:


def plot_predictions(d, forecasts, sd_estimate, window_size, limit_begin = 0, limit_end = None, ax = None, legend = True, look_ahead_window = 1):
    if limit_end is None:
        observed = np.array(d[(window_size + look_ahead_window - 1 + limit_begin):])
    else:
        observed = np.array(d[(window_size + look_ahead_window - 1 + limit_begin):(window_size + look_ahead_window -1 + limit_end)])
    if ax is None:
        plt.plot(observed, '.', label = 'observed')
        plt.plot(forecasts[limit_begin:limit_end], '.', label = 'predicted')
        plt.plot(forecasts[limit_begin:limit_end] + 2*sd_estimate, '-', label = "prediction + 2sd")
        plt.plot(forecasts[limit_begin:limit_end] - 2*sd_estimate, '-', label = "predictionn - 2sd")
        if legend:
            plt.legend()
    else:
        ax.plot(observed, '.', label = 'observed')
        ax.plot(forecasts[limit_begin:limit_end], '.', label = 'predicted')
        ax.plot(forecasts[limit_begin:limit_end] + 2*sd_estimate, '-', label = "prediction + 2sd")
        ax.plot(forecasts[limit_begin:limit_end] - 2*sd_estimate, '-', label = "predictionn - 2sd")
        if legend:
            ax.legend()


# In[188]:




