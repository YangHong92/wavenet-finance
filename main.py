import pandas as pd
import numpy as np
from myutil import calculate_receptive_field, get_bin_border_with_equal_count
from generative_wavenet import EnhancedBasicWaveNet

import tensorflow as tf
config = tf.ConfigProto() 
config.gpu_options.allow_growth=True 
sess = tf.Session(config=config)

import keras.backend.tensorflow_backend as KTF
KTF.set_session(sess)

# ********* parepare data *********
SP500 = pd.read_csv('./data/SP500_05-16.csv', parse_dates=['Date'], usecols=['Date','Price']).rename(columns={'Price':'SP500'})
Volty = pd.read_csv('./data/Volatility_05-16.csv', parse_dates=['Date'], usecols=['Date','Price']).rename(columns={'Price':'Volty'})
Volty = Volty.sort_values(['Date'], ascending=True)

merged_ft = pd.merge(SP500, Volty, on=['Date'], how='inner')
print('Shape of merged data from SP500 and Volatility index', merged_ft.shape)

merged_ft['SPX_rtn'] = merged_ft['SP500'].pct_change().fillna(0)

FTSE = pd.read_csv('./data/FTSE100_05-16.csv', parse_dates=['Date'], dtype={'FTSE':float}, usecols=['Date','Price']).rename(columns={'Price':'FTSE'})
FTSE = FTSE.sort_values(['Date'], ascending=True)

GBP_USD = pd.read_csv('./data/GBP_USD 05-16.csv', parse_dates=['Date'], usecols=['Date','Price']).rename(columns={'Price':'GBP_USD'})
GBP_USD = GBP_USD.sort_values(['Date'], ascending=True)

dataset = pd.merge(pd.merge(merged_ft, FTSE, on=['Date'], how='inner'), GBP_USD, on=['Date'], how='inner')

print('Shape of merged data from SP500, Volatility index, TFSE100, GBP_USD', dataset.shape)
print('Records decrease as there are some non overlaping trading dates for US and UK market')
dataset.head()

X = dataset.loc[:,['SP500','FTSE', 'GBP_USD']].values 
y = dataset.loc[:,['SP500']].values 
print(X.shape)

# ********* setting parameter *********
solution = {0:'classification', 1:'regression'}
quantization_channels = 80
test_round = 500
batch_size = 100
epochs = 50
num_blocks = 3
num_layers = 6 
num_hidden = 128
filter_width = 2
dilations = [2**i for i in range(num_layers)] * num_blocks
receptive_field = calculate_receptive_field(dilations, filter_width)  
print("receptive_field: ", receptive_field)
SPX = X[:,0]

# ********* equal bin_width *********
bins = np.linspace(round(SPX.min(), 4)-0.0001, round(SPX.max(), 4)+0.0001, num = quantization_channels+1) 
y_discret = np.digitize(SPX, bins) - 1

ehwavenet = EnhancedBasicWaveNet(num_time_samples = receptive_field, 
      num_classes = quantization_channels, 
      use_condition = False, # no global_condition
      num_channels = X.shape[-1], 
      num_blocks = num_blocks, 
      num_layers = num_layers, 
      num_hidden = num_hidden, 
      use_residual = True,
      use_skip = True,                        
      solution = solution[0])

ehwavenet.iterative_train(X, y_discret, test_round = test_round, batch_size=batch_size, epochs=epochs, y_feature_axis_in_X=0, should_norm_y=False, weight_file="itr_eqb_res_skip_wt.h5")


ehwavenet = EnhancedBasicWaveNet(num_time_samples = receptive_field, 
      num_classes = quantization_channels, 
      use_condition = True, # with global_condition
      num_channels = X.shape[-1], 
      num_blocks = num_blocks, 
      num_layers = num_layers, 
      num_hidden = num_hidden, 
      use_residual = True,
      use_skip = True,                        
      solution = solution[0])

ehwavenet.iterative_train(X, y_discret, test_round = test_round, batch_size=batch_size, epochs=epochs, y_feature_axis_in_X=0, should_norm_y=False, weight_file="itr_eqb_res_skip_cond_wt.h5")
# ********* equal bin_width *********


# ********* unequal bin_width *********
uneq_bin_borders = get_bin_border_with_equal_count(SPX, quantization_channels)
y_uneq_discret = np.digitize(SPX, uneq_bin_borders) - 1

ehwavenet = EnhancedBasicWaveNet(num_time_samples = receptive_field, 
      num_classes = quantization_channels, 
      use_condition = False, # no global_condition
      num_channels = X.shape[-1], 
      num_blocks = num_blocks, 
      num_layers = num_layers, 
      num_hidden = num_hidden,
      use_residual = True,
      use_skip = True,                        
      solution = solution[0])

ehwavenet.iterative_train(X, y_uneq_discret, test_round = test_round, batch_size=batch_size, epochs=epochs, y_feature_axis_in_X=0, should_norm_y=False, weight_file="itr_uneqb_res_skip_wt.h5")


ehwavenet = EnhancedBasicWaveNet(num_time_samples = receptive_field, 
      num_classes = quantization_channels, 
      use_condition = True, # with global_condition
      num_channels = X.shape[-1], 
      num_blocks = num_blocks, 
      num_layers = num_layers, 
      num_hidden = num_hidden, 
      use_residual = True,
      use_skip = True,                        
      solution = solution[0])

ehwavenet.iterative_train(X, y_uneq_discret, test_round = test_round, batch_size=batch_size, epochs=epochs, y_feature_axis_in_X=0, should_norm_y=False, weight_file="itr_uneqb_res_skip_cond_wt.h5")
# ********* unequal bin_width *********



