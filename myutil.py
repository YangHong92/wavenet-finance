import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from datetime import timedelta
import math
from functools import reduce

def calculate_receptive_field(dilations, filter_width):
    # return 2 ** (len(dilations) - 1) * filter_width
    return (filter_width - 1) * sum(dilations) + 1 # rf=15 when dilations=[1,2,4,1,2,4]

def calc_match_rate(pred_outs, y_test): 
    return np.where(pred_outs == y_test, 1, 0).sum() / pred_outs.shape[0]

def calc_mean_abs_diff(pred_outs, y_test): 
    return np.mean(np.absolute(pred_outs - y_test))

def get_bin_border_with_equal_count(data, n_bins):
    data = np.sort(data)
    n_values = data.shape[0]
    bin_borders = [round(data.min(), 4)-0.0001] + [data[(n_values // n_bins) * i] for i in range(1, n_bins)] + [round(data.max(), 4)+0.0001] 
    
    return bin_borders

def calc_bins(data, quantization_channels):
    bin_min = round(data.min(), 4)-0.0001
    bin_max = round(data.max(), 4)+0.0001

    bins = np.linspace(bin_min, bin_max, num = quantization_channels+1) #bin_max is included
    # bin_unit = bins[1] - bins[0]
    # bins = np.arange(bin_min, bin_max, bin_unit) #bin_max is not included
    hist, _ = np.histogram(data, bins = bins) 

    bins_items = []
    def combine_adjacent_bins(prev_, next_):
        bins_items.append((str(prev_)+'~'+str(next_)))
        return next_
    reduce(combine_adjacent_bins, bins)
    bins_items = np.array(bins_items)
    print('bins_items[-5:]: ', bins_items[-5:], bins_items.shape)
    print("0 count bins: ", bins_items[np.where(hist==0, True, False)].shape)

    plt.hist(data, bins = bins) 
    plt.title("histogram for SPX") 
    plt.show()
    
    return bins

def generateLorenz(x=0.0, y=1.0, z=1.05, dt=0.01, stepCnt=3000, s=10, r=28, b=2.667, figure=False):
    
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot
    
    # Need one more for the initial values
    xs = np.empty((stepCnt + 1,))
    ys = np.empty((stepCnt + 1,))
    zs = np.empty((stepCnt + 1,))

    # Setting initial values
    xs[0], ys[0], zs[0] = (x, y, z)
    
    # Stepping through "time".
    for i in range(stepCnt):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    if figure:    
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(xs, ys, zs, lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")
        
        plt.show()
    
    return xs,ys,zs

class Normalizer(object):
    """ Simple object for doing a mean-centering stddev scaling """

    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, array, axis):
        """ Fit based on all values in array """
        self.mean = np.mean(array, axis=axis, dtype=np.float32)
        self.std = np.std(array, axis=axis, dtype=np.float32) + 1e-8
        return self
    
    def transform(self, array, y_feature_axis_in_X=None):
        """ Transform array according to the mean found in .fit() """ 
        if not self.mean.all():
            raise IOError("You must call .fit() before .transform()")
        if y_feature_axis_in_X != None:
            return (array - self.mean[y_feature_axis_in_X]) / self.std[y_feature_axis_in_X]
        else:
            return (array - self.mean) / self.std

    def inverse_transform(self, array, y_feature_axis_in_X=None):
        """ Undo .transform """
        if not self.mean.all():
            raise IOError("You must call .fit() before .inverse_transform()")
        if y_feature_axis_in_X != None:
            return array * self.std[y_feature_axis_in_X] + self.mean[y_feature_axis_in_X]
        else:
            return array * self.std + self.mean

    def fit_transform(self, array, axis=0):
        """ Call .fit() -> .transform() """
        self.fit(array, axis)
        return self.transform(array)

def split_batch_norm_NXNy(normalizer, split_rate, X, y, y_feature_axis_in_X, should_norm_y, receptive_field):
    
    batches = X.shape[0] - receptive_field
    
    def new_prepare_batch(X, y, batches, receptive_field):
    
        output_width = receptive_field
        # input_width = output_width + sum(dilations)
        input_width = output_width
        
        # X_new.shape : [n_samples (n_batches), following_based_time_steps+padded, features (channels)]
        X_new = np.zeros((batches, input_width, X.shape[1] if len(X.shape) > 1 else 1))
        y_new = np.zeros((batches, output_width, 1))
        for ix in range(X_new.shape[0]): #ix is the number of batch
            for jx in range(receptive_field): #jx is the following time steps that are based to make prediction
                X_new[ix, jx, :] = X[ix + jx, :] if len(X.shape) > 1 else X[ix + jx] # jx = 0 ~ (receptive_field-1)
                y_new[ix, jx, :] = y[ix + jx + 1]  
        return X_new, y_new

    if split_rate < 1:
        split = int(split_rate*batches)
    else:
        split = batches - 1
    
    X_batch, y_batch = new_prepare_batch(X, y, batches, receptive_field)
    
    if(normalizer is not None):
        X_train = X[:split] 
        _ = normalizer.fit_transform(X_train)
        normed_X_batch = normalizer.transform(X_batch)
        if should_norm_y:
          normed_y_batch = normalizer.transform(y_batch, y_feature_axis_in_X)
        else:
          normed_y_batch = y_batch
        
        X_train = normed_X_batch[:split]
        X_test = normed_X_batch[split:]

        y_train = normed_y_batch[:split]
        y_test = normed_y_batch[split:]
    else:
        X_train = X_batch[:split]
        X_test = X_batch[split:]

        y_train = y_batch[:split]
        y_test = y_batch[split:]
    
    return X_train, y_train, X_test, y_test

def split_batch_norm_X_multi_y(normalizer, split_rate, X, y, y_feature_axis_in_X, should_norm_y, no_to_predict_y, receptive_field):
    """ prepare batch by sliding window from time series data """
    sample_width = receptive_field + no_to_predict_y-1
    
    # calculate number of batches
    batches = int((X.shape[0] - no_to_predict_y - sample_width)/no_to_predict_y)+1
    
    def prepare_batch_multi_y(X, y, sample_width, batches, no_to_predict_y, receptive_field):
        # the number of (sample_width - receptive_field + 1) is the number of output to be forecasted
        output_width = no_to_predict_y 
        input_width = sample_width
        
        # X_new.shape : [n_samples (n_batches), based_time_steps, features (channels)]
        X_new = np.zeros((batches, input_width, X.shape[1]))
        y_new = np.zeros((batches, output_width, 1))
        for ix in range(X_new.shape[0]): #ix is the number of batch
            for jx in range(sample_width): #jx is the following time steps to be based for prediction
                X_new[ix, jx, :] = X[ix*no_to_predict_y + jx, :]
                if jx < output_width: 
                    y_new[ix, jx, :] = y[ix*no_to_predict_y + sample_width + jx]
        return X_new, y_new

    if split_rate < 1:
        split = int(split_rate*batches)
    else:
        split = batches - 1  

    X_batch, y_batch = prepare_batch_multi_y(X, y, sample_width, batches, no_to_predict_y, receptive_field)
    
    if(normalizer is not None):
        X_train = X[:split]
        _ = normalizer.fit_transform(X_train)
        normed_X_batch = normalizer.transform(X_batch)
        if should_norm_y:
          normed_y_batch = normalizer.transform(y_batch, y_feature_axis_in_X)
        else:
          normed_y_batch = y_batch
          
        X_train = normed_X_batch[:split]
        X_test = normed_X_batch[split:]

        y_train = normed_y_batch[:split]
        y_test = normed_y_batch[split:]
    else:
        X_train = X_batch[:split]
        X_test = X_batch[split:]

        y_train = y_batch[:split]
        y_test = y_batch[split:]
    
    return X_train, y_train, X_test, y_test
  
def split_batch_norm_X_y(normalizer, split_rate, X, y, y_feature_axis_in_X, time_steps=100):
    
    batches = X.shape[0] - time_steps
    
    def prepare_batch(X, y, time_steps=100):
        # X_new.shape : [n_samples (n_batches), following_based_time_steps, features (channels)]
        X_new = np.zeros((batches, time_steps, X.shape[1]))
        y_new = np.zeros((batches, 1))
        for ix in range(X_new.shape[0]): #ix is the number of batch
            for jx in range(time_steps): #jx is the following time steps that are based to make prediction
                X_new[ix, jx, :] = X[ix + jx, :] # jx=0-99
            y_new[ix] = y[ix + time_steps] # time_steps=100    
        return X_new, y_new

    if split_rate < 1:
        split = int(split_rate*batches)
    else:
        split = batches - 1

    X_batch, y_batch = prepare_batch(X, y, time_steps)
    
    if(normalizer is not None):
        X_train = X[:split]
        _ = normalizer.fit_transform(X_train)
        normed_X_batch = normalizer.transform(X_batch)
        normed_y_batch = normalizer.transform(y_batch, y_feature_axis_in_X)
        
        X_train = normed_X_batch[:split]
        X_test = normed_X_batch[split:]

        y_train = normed_y_batch[:split]
        y_test = normed_y_batch[split:]

        # normed_X = normalizer.fit_transform(X)
        # normed_y = normalizer.transform(y, y_feature_axis_in_X)
        # X_batch, y_batch = prepare_batch(normed_X, normed_y, time_steps)
        # X_train = X_batch[:split]
        # X_test = X_batch[split:]

        # y_train = y_batch[:split]
        # y_test = y_batch[split:]
    else:
        X_train = X_batch[:split]
        X_test = X_batch[split:]

        y_train = y_batch[:split]
        y_test = y_batch[split:]
    
    return X_train, y_train, X_test, y_test

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true), axis=-1)

def visualize_forecast_plot(pred_outs_back, y_test_back, show=True, save_figure=False, figname=None):
    if len(pred_outs_back.shape) > 1:
        pred_outs_back = pred_outs_back.reshape(-1)
    if len(y_test_back.shape) > 1:
        y_test_back = y_test_back.reshape(-1)

    df_pred_outs_back = pd.Series(pred_outs_back)
    df_y_test_back = pd.Series(y_test_back)

    fig = plt.figure(figsize=(12, 8))
    df_pred_outs_back.plot(label='Forecast', alpha=.75)
    df_y_test_back.plot(label='Actual', alpha=.75, ls='--')
    plt.legend(fontsize=12)
    plt.xlabel("Time steps")
    plt.ylabel("Price")
    if figname is not None:
        plt.title(figname.split('.')[0])
    if save_figure and figname is not None:
        plt.savefig(figname)
    if show:
        plt.show()
    else:
        plt.close(fig)

def visualize_forecast_scatter(pred_outs_back, y_test_back, show=True, save_figure=False, figname=None):
    if len(pred_outs_back.shape) > 1:
        pred_outs_back = pred_outs_back.reshape(-1)
    if len(y_test_back.shape) > 1:
        y_test_back = y_test_back.reshape(-1)

    df_pred_outs_back = pd.Series(pred_outs_back)
    df_y_test_back = pd.Series(y_test_back)

    fig = plt.figure(figsize=(12, 8))
    plt.scatter(np.arange(0, df_pred_outs_back.shape[0]), df_pred_outs_back, label='Forecast', alpha=.75, marker='.')
    plt.scatter(np.arange(0, df_y_test_back.shape[0]), df_y_test_back, label='Actual', alpha=.75, marker='.')
    plt.legend(fontsize=12)
    plt.xlabel("Time steps")
    plt.ylabel("Price")
    if figname is not None:
        plt.title(figname.split('.')[0])
    if save_figure and figname is not None:
        plt.savefig(figname)
    if show:
        plt.show()
    else:
        plt.close(fig)

def encode_in_bins(bins, arr):
    # return np.array(list(map(lambda x: math.floor((x - bin_min)/bin_unit), arr)))
    # right excluded: bins[i-1] <= x < bins[i]
    return np.digitize(arr, bins, right=False) - 1 
    
def decode_from_bins(bins, arr):
    #return np.array(list(map(lambda x: (x*bin_unit + bin_min), arr)))
    return bins[arr]

def inverse_pct_change(arr, price_0):
    """ convert daily return back to prices"""
    price = [] 
    for i in range(arr.shape[0]):
        price.append(arr[i] * price_0 + price_0) if i==0 else price.append(arr[i] * price[-1] + price[-1])
    return np.array(price)

def smape(forecast, actual, i, normalizer=None):
    """ Symmetric Mean Absolute Percent Error """
    forecast = np.reshape(forecast, (-1,))
    actual = np.reshape(actual, (-1,))

    if normalizer:
        forecast = normalizer.inverse_transform(forecast, i)
        actual = normalizer.inverse_transform(actual, i)

    N = len(forecast)
    return 200 / N * np.sum(np.abs(forecast - actual) / (np.abs(forecast) + np.abs(actual)))

def walk_forward_split(time_data, pred_steps=1000):
    try:
        if 'date' in time_data.columns:
            first_day = time_data.iloc[0].date 
            last_day = time_data.iloc[-1].date

            pred_length=timedelta(pred_steps)

            val_pred_start = last_day - pred_length + timedelta(1) # date back (1000-1) days, so 1000 days self-included
            val_pred_end = last_day

            train_pred_start = val_pred_start - pred_length
            train_pred_end = val_pred_start - timedelta(1)

            train_enc_start = first_day
            train_enc_end =  train_pred_start - timedelta(1)

            enc_length = train_enc_end -  train_enc_start

            val_enc_end = val_pred_start - timedelta(1)
            val_enc_start = val_enc_end - enc_length

            print('Train encoding:', train_enc_start, '~', train_enc_end)
            print('Train prediction:', train_pred_start, '~', train_pred_end, '\n')
            print('Val encoding:', val_enc_start, '~', val_enc_end)
            print('Val prediction:', val_pred_start, '~', val_pred_end)

            print('\nEncoding interval:', enc_length.days)
            print('Prediction interval:', pred_length.days)

            date_to_index = pd.Series(index=pd.Index([c for c in time_data[0:].date]), data=[i for i in range(len(time_data[0:]))])

            train_enc_index = date_to_index[train_enc_start:train_enc_end]
            train_pred_index = date_to_index[train_pred_start:train_pred_end]
            val_enc_index = date_to_index[val_enc_start:val_enc_end]
            val_pred_index = date_to_index[val_pred_start:val_pred_end]
            
            return time_data.values[train_enc_index], time_data.values[train_pred_index], time_data.values[val_enc_index], time_data.values[val_pred_index]
        else:    
            raise Exception('')
    except:
        print("dataframe should have 'date' column")

def split_data(df, test_period=360):
    train_features = df.iloc[:-test_period]
    
    #value move forward, index(timestamp) remain
    train_targets = df.shift(-1).iloc[:-test_period]
    test_targets = df.shift(-1).iloc[-test_period:-1]
    
    return train_features, train_targets, test_targets