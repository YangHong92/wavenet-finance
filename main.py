import pandas as pd
import numpy as np
from myutil import calculate_receptive_field, get_bin_border_with_equal_count, mean_absolute_error
from generative_wavenet import EnhancedBasicWaveNet
import json

# import tensorflow as tf
# config = tf.ConfigProto() 
# config.gpu_options.allow_growth=True 
# sess = tf.Session(config=config)

# import keras.backend.tensorflow_backend as KTF
# KTF.set_session(sess)

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
X_ewm = dataset.loc[:,['SP500','FTSE', 'GBP_USD']].ewm(span=20).mean().values 
y = dataset.loc[:,['SP500']].values 
print(X.shape, X_ewm.shape)

# ********* setting parameter *********
solution = {0:'classification', 1:'regression'}
quantization_channels = 80

with open("args.json", "r") as jsfile:
      args = json.load(jsfile)
      test_round = int(args["test_round"])
      batch_size = int(args["batch_size"])
      epochs = int(args["epochs"])
      test_step = int(args["test_step"])
      run_example = np.array(args["run_example"])

num_blocks = 2
num_layers = 5 
num_hidden = 128
filter_width = 2
dilations = [2**i for i in range(num_layers)] * num_blocks
receptive_field = calculate_receptive_field(dilations, filter_width)  
print("receptive_field: ", receptive_field)
SPX = X[:,0]
SPX_ewm = X_ewm[:,0]

with open("mean_absolute_errors.txt", "w") as f:

      # ********* equal bin_width *********
      bins = np.linspace(round(SPX.min(), 4)-0.0001, round(SPX.max(), 4)+0.0001, num = quantization_channels+1) 
      y_discret = np.digitize(SPX, bins) - 1

      if 0 in run_example:
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

            targets, preds = ehwavenet.iterative_train(X, y_discret, test_round = test_round, batch_size=batch_size, epochs=epochs, y_feature_axis_in_X=0, should_norm_y=False, weight_file="itr_eqb_res_skip_wt.h5")
            f.writelines("%s\n" % item for item in preds)
            f.write("itr_eqb_res_skip_predict: " + str(mean_absolute_error(targets, preds)) + "\n") 
            peds = ehwavenet.generate(X[-test_round-1,0][None,None], X[-test_round-1,1][None,None], test_round, y_discret[-test_round:], "itr_eqb_res_skip_wt.h5")
            f.writelines("%s\n" % item for item in peds)
            f.write("itr_eqb_res_skip_generate: " + str(mean_absolute_error(y_discret[-test_round:], peds)) + "\n")

      if 1 in run_example:
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

            targets, preds = ehwavenet.iterative_train(X, y_discret, test_round = test_round, batch_size=batch_size, epochs=epochs, y_feature_axis_in_X=0, should_norm_y=False, weight_file="itr_eqb_res_skip_cond_wt.h5")
            f.writelines("%s\n" % item for item in preds)
            f.write("itr_eqb_res_skip_cond_predict: " + str(mean_absolute_error(targets, preds)) + "\n")
            peds = ehwavenet.generate(X[-test_round-1,0][None,None], X[-test_round-1,1][None,None], test_round, y_discret[-test_round:], "itr_eqb_res_skip_cond_wt.h5")
            f.writelines("%s\n" % item for item in peds)
            f.write("itr_eqb_res_skip_cond_generate: " + str(mean_absolute_error(y_discret[-test_round:], peds)) + "\n")

      if 2 in run_example:
            ehwavenet = EnhancedBasicWaveNet(num_time_samples = receptive_field, 
                  num_classes = quantization_channels, 
                  use_condition = False, # no global_condition
                  num_channels = X.shape[-1], 
                  num_blocks = num_blocks, 
                  num_layers = num_layers, 
                  num_hidden = num_hidden, 
                  use_skip = True,
                  solution = solution[0])

            targets, preds = ehwavenet.iterative_train(X, y_discret, test_round = test_round, batch_size=batch_size, epochs=epochs, y_feature_axis_in_X=0, should_norm_y=False, weight_file="itr_eqb_skip_wt.h5")
            f.writelines("%s\n" % item for item in preds)
            f.write("itr_eqb_skip_predict: " + str(mean_absolute_error(targets, preds)) + "\n")
            peds = ehwavenet.generate(X[-test_round-1,0][None,None], X[-test_round-1,1][None,None], test_round, y_discret[-test_round:], "itr_eqb_skip_wt.h5")
            f.writelines("%s\n" % item for item in peds)
            f.write("itr_eqb_skip_generate: " + str(mean_absolute_error(y_discret[-test_round:], peds)) + "\n")

      if 3 in run_example:
            ehwavenet = EnhancedBasicWaveNet(num_time_samples = receptive_field, 
                  num_classes = quantization_channels, 
                  use_condition = False, # no global_condition
                  num_channels = X.shape[-1], 
                  num_blocks = num_blocks, 
                  num_layers = num_layers, 
                  num_hidden = num_hidden, 
                  use_residual = True,
                  solution = solution[0])

            targets, preds = ehwavenet.iterative_train(X, y_discret, test_round = test_round, batch_size=batch_size, epochs=epochs, y_feature_axis_in_X=0, should_norm_y=False, weight_file="itr_eqb_res_wt.h5")
            f.writelines("%s\n" % item for item in preds)
            f.write("itr_eqb_res_predict: " + str(mean_absolute_error(targets, preds)) + "\n")
            peds = ehwavenet.generate(X[-test_round-1,0][None,None], X[-test_round-1,1][None,None], test_round, y_discret[-test_round:], "itr_eqb_res_wt.h5")
            f.writelines("%s\n" % item for item in peds)
            f.write("itr_eqb_res_generate: " + str(mean_absolute_error(y_discret[-test_round:], peds)) + "\n")
      # ********* equal bin_width *********


      # ********* unequal bin_width *********
      uneq_bin_borders = get_bin_border_with_equal_count(SPX, quantization_channels)
      y_uneq_discret = np.digitize(SPX, uneq_bin_borders) - 1

      if 4 in run_example:
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

            targets, preds = ehwavenet.iterative_train(X, y_uneq_discret, test_round = test_round, batch_size=batch_size, epochs=epochs, y_feature_axis_in_X=0, should_norm_y=False, weight_file="itr_uneqb_res_skip_wt.h5")
            f.writelines("%s\n" % item for item in preds)
            f.write("itr_uneqb_res_skip_predict: " + str(mean_absolute_error(targets, preds)) + "\n")
            peds = ehwavenet.generate(X[-test_round-1,0][None,None], X[-test_round-1,1][None,None], test_round, y_uneq_discret[-test_round:], "itr_uneqb_res_skip_wt.h5")
            f.writelines("%s\n" % item for item in peds)
            f.write("itr_uneqb_res_skip_generate: " + str(mean_absolute_error(y_uneq_discret[-test_round:], peds)) + "\n")

      if 5 in run_example:
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

            targets, preds = ehwavenet.iterative_train(X, y_uneq_discret, test_round = test_round, batch_size=batch_size, epochs=epochs, y_feature_axis_in_X=0, should_norm_y=False, weight_file="itr_uneqb_res_skip_cond_wt.h5")
            f.writelines("%s\n" % item for item in preds)
            f.write("itr_uneqb_res_skip_cond_predict: " + str(mean_absolute_error(targets, preds)) + "\n")
            peds = ehwavenet.generate(X[-test_round-1,0][None,None], X[-test_round-1,1][None,None], test_round, y_uneq_discret[-test_round:], "itr_uneqb_res_skip_cond_wt.h5")
            f.writelines("%s\n" % item for item in peds)
            f.write("itr_uneqb_res_skip_cond_generate: " + str(mean_absolute_error(y_uneq_discret[-test_round:], peds)) + "\n")
      # ********* unequal bin_width *********

      # ********* unequal bin_width + iterative_step_train  *********
      if 6 in run_example:
            uneq_ehwavenet = EnhancedBasicWaveNet(num_time_samples = receptive_field, 
                  num_classes = quantization_channels, 
                  use_condition = True, # with global_condition
                  num_channels = X.shape[-1],                              
                  num_blocks = num_blocks, 
                  num_layers = num_layers, 
                  num_hidden = num_hidden,
                  use_skip = True,
                  use_residual = True,                              
                  solution = solution[0])

            targets, preds = uneq_ehwavenet.iterative_step_train(X, y_uneq_discret, test_round=test_round, batch_size=batch_size, epochs=epochs, test_step=test_step, y_feature_axis_in_X=0, should_norm_y=False, weight_file="uneq_res_skip_cond_step_wt.h5")
            f.writelines("%s\n" % item for item in preds)
            f.write("uneq_res_skip_cond_step_predict: " + str(mean_absolute_error(targets, preds)) + "\n")
            peds = uneq_ehwavenet.generate(X[-test_round*test_step-1,0][None,None], X[-test_round*test_step-1,1][None,None], test_round*test_step, y_uneq_discret[-test_round*test_step:], "uneq_res_skip_cond_step_wt.h5")
            f.writelines("%s\n" % item for item in peds)
            f.write("uneq_res_skip_cond_step_generate: " + str(mean_absolute_error(y_uneq_discret[-test_round*test_step:], peds)) + "\n")
      # ********* unequal bin_width + iterative_step_train  *********

      # ********* unequal bin_width + moving average *********
      uneq_bin_borders_ewm = get_bin_border_with_equal_count(SPX_ewm, quantization_channels)
      y_uneq_discret_ewm = np.digitize(SPX_ewm, uneq_bin_borders_ewm) - 1

      if 7 in run_example:
            uneq_ehwavenet = EnhancedBasicWaveNet(num_time_samples = receptive_field, 
                  num_classes = quantization_channels, 
                  use_condition = True, # with global_condition
                  num_channels = X_ewm.shape[-1],                              
                  num_blocks = num_blocks, 
                  num_layers = num_layers, 
                  num_hidden = num_hidden,
                  use_skip = True,
                  use_residual = True,                              
                  solution = solution[0])

            targets, preds = uneq_ehwavenet.iterative_train(X_ewm, y_uneq_discret_ewm, test_round = test_round, batch_size=batch_size, epochs=epochs, y_feature_axis_in_X=0, should_norm_y=False, weight_file="uneq_ewm_res_skip_cond_wt.h5")
            f.writelines("%s\n" % item for item in preds)
            f.write("uneq_ewm_res_skip_cond_predict: " + str(mean_absolute_error(targets, preds)) + "\n")
            peds = uneq_ehwavenet.generate(X_ewm[-test_round-1,0][None,None], X_ewm[-test_round-1,1][None,None], test_round, y_uneq_discret_ewm[-test_round:], "uneq_ewm_res_skip_cond_wt.h5")
            f.writelines("%s\n" % item for item in peds)
            f.write("uneq_ewm_res_skip_cond_generate: " + str(mean_absolute_error(y_uneq_discret_ewm[-test_round:], peds)) + "\n")
      # ********* unequal bin_width + moving average *********

      # ********* unequal bin_width + moving average + iterative_step_train *********
      if 8 in run_example:
            uneq_ehwavenet = EnhancedBasicWaveNet(num_time_samples = receptive_field, 
                  num_classes = quantization_channels, 
                  use_condition = True, # with global_condition
                  num_channels = X_ewm.shape[-1],                              
                  num_blocks = num_blocks, 
                  num_layers = num_layers, 
                  num_hidden = num_hidden,
                  use_skip = True,
                  use_residual = True,                              
                  solution = solution[0])

            targets, preds = uneq_ehwavenet.iterative_step_train(X_ewm, y_uneq_discret_ewm, test_round=test_round, batch_size=batch_size, epochs=epochs, test_step=test_step, y_feature_axis_in_X=0, should_norm_y=False, weight_file="uneq_ewm_res_skip_cond_step_wt.h5")
            f.writelines("%s\n" % item for item in preds)
            f.write("uneq_ewm_res_skip_cond_step_predict: " + str(mean_absolute_error(targets, preds)) + "\n")
            peds = uneq_ehwavenet.generate(X_ewm[-test_round*test_step-1,0][None,None], X_ewm[-test_round*test_step-1,1][None,None], test_round*test_step, y_uneq_discret_ewm[-test_round*test_step:], "uneq_ewm_res_skip_cond_step_wt.h5")
            f.writelines("%s\n" % item for item in peds)
            f.write("uneq_ewm_res_skip_cond_step_generate: " + str(mean_absolute_error(y_uneq_discret_ewm[-test_round*test_step:], peds)) + "\n")
      # ********* unequal bin_width + moving average + iterative_step_train *********
     
