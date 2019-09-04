import numpy as np
from myutil import Normalizer, split_batch_norm_NXNy, visualize_forecast_plot, visualize_forecast_scatter

import tensorflow as tf
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv1D, Input, Activation, Multiply, Add, Concatenate, ZeroPadding1D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
import os
from keras import backend as K

def generator(X_data, y_data, batch_size):
 
    number_of_batches = X_data.shape[0]/batch_size  
    counter=0

    while True:

        X_batch = X_data[batch_size*counter:batch_size*(counter+1), :, :]
        y_batch = y_data[batch_size*counter:batch_size*(counter+1), :, :]
        counter += 1

        yield (X_batch, y_batch)

        #restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0

class EnhancedBasicWaveNet(object):
    """
      input = [None, num_time_samples, num_channels]
      output = [None, num_time_samples, num_classes]
    """
    def __init__(self,
                 num_time_samples,
                 num_channels=1,
                 num_classes=80,
                 num_blocks=2,
                 num_layers=4,
                 num_hidden=128,
                 use_condition = False,
                 use_residual=False,
                 use_skip=False,
                 solution='classification'):
        
        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.use_condition = use_condition
        self.use_residual = use_residual
        self.use_skip = use_skip
        self.solution = solution
     
    def create_network_loss(self, input_batch):
        
        if input_batch.shape[-1] > 1 and self.use_condition:
            x = Lambda(lambda x: tf.expand_dims(x[:, :, 0], axis=-1))(input_batch)
            global_condition = Lambda(lambda x: tf.expand_dims(x[:, :, 1], axis=-1))(input_batch)
            print("with condition")
        else:
            x = Lambda(lambda x: tf.expand_dims(x[:, :, 0], axis=-1))(input_batch)
            global_condition = None
            print("without condition")

        h = self.conv1d(x,
                       self.num_hidden, # self.residual_channels,
                       filter_width=1,
                       gain=1.0,
                       activation='relu',
                       bias=True,
                       name='ipt-1x1')

        skips = []
        
        for b in range(self.num_blocks):
            for i in range(self.num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                if self.use_skip:
                    skip, h = self.dilated_conv1d(h, self.num_hidden, rate=rate, global_condition=global_condition, name=name)          
                    skips.append(skip)                    
                else:
                    h = self.dilated_conv1d(h, self.num_hidden, rate=rate, global_condition=global_condition, name=name)
                
         
        if self.solution == 'classification':
            if self.use_skip:
                out = Activation('relu')(Add()(skips))
                predictions = self.conv1d(out,
                                 self.num_classes,
                                 filter_width=1,
                                 gain=1.0,
                                 activation='softmax',
                                 bias=True,
                                 name='f-1x1')
            else:
                predictions = self.conv1d(h,
                                 self.num_classes,
                                 filter_width=1,
                                 gain=1.0,
                                 activation='softmax',
                                 bias=True,
                                 name='f-1x1')
        else:
            if self.use_skip:
                out = Activation('relu')(Add()(skips))
                predictions = self.conv1d(out,
                                 1,
                                 filter_width=1,
                                 gain=1.0,
                                 activation='linear',
                                 bias=True,
                                 name='f-1x1')
            else:
                predictions = self.conv1d(h,
                                 1,
                                 filter_width=1,
                                 gain=1.0,
                                 activation='linear',
                                 bias=True,
                                 name='f-1x1')
        return predictions
    
    def conv1d(self,
           inputs,
           out_channels,
           filter_width=2,
           padding='valid',
           name=None,
           dilation_rate=1,
           gain=np.sqrt(2),
           batch_norm=False,
           bias=True,
           activation='relu'):
        
  
        in_channels = inputs.get_shape().as_list()[-1]
        stddev = gain / np.sqrt(filter_width**2 * in_channels)

        outputs = Conv1D(filters=out_channels, 
                       kernel_size=filter_width, 
                       padding=padding, 
                       dilation_rate=dilation_rate, 
                       kernel_initializer=tf.random_normal_initializer(stddev=stddev), 
                       use_bias=bias,
                       name=name)(inputs)

        if batch_norm:
            outputs = BatchNormalization()(outputs)

        if activation:
            outputs = Activation(activation)(outputs)

        return outputs
  
    def dilated_conv1d(self,
                   inputs,
                   out_channels,
                   filter_width=2,
                   rate=1,
                   global_condition=None,
                   padding='valid',
                   bias=True,
                   name=None,
                   gain=np.sqrt(2),
                   activation='relu'):
      
        conv_filter = self.conv1d(inputs,
                        out_channels=out_channels,
                        filter_width=filter_width, # dilated_conv 
                        padding=padding,
                        bias=bias,
                        name=name+'-filter-x',
                        dilation_rate=rate,
                        gain=gain,
                        activation=None)
        
        conv_gate = self.conv1d(inputs,
                        out_channels=out_channels,
                        filter_width=filter_width, # dilated_conv
                        padding=padding,
                        bias=bias,
                        name=name+'-gate-x',
                        dilation_rate=rate,
                        gain=gain,
                        activation=None)
        
        conv_filter = ZeroPadding1D( (rate*(filter_width-1), 0) )(conv_filter)
        conv_gate = ZeroPadding1D( (rate*(filter_width-1), 0) )(conv_gate)
        
        # add linear projection of global condition 
        if global_condition is not None:
            conv_filter = Add()([conv_filter, self.conv1d(global_condition,
                                        out_channels=out_channels,
                                        filter_width=1, # 1x1_conv
                                        padding=padding,
                                        bias=bias,
                                        name=name+'-filter-c',
                                        dilation_rate=rate,
                                        gain=gain,
                                        activation=None)])
            conv_gate = Add()([conv_gate, self.conv1d(global_condition,
                                        out_channels=out_channels,
                                        filter_width=1, # 1x1_conv
                                        padding=padding,
                                        bias=bias,
                                        name=name+'-gate-c',
                                        dilation_rate=rate,
                                        gain=gain,
                                        activation=None)])

        # add activation tanh, sigmoid, and multiply
        outputs = Multiply()([Activation('tanh')(conv_filter), Activation('sigmoid')(conv_gate)])
     
        if self.use_skip:
            skip = self.conv1d(outputs,
                             out_channels=self.num_hidden, # self.skip_channels,
                             filter_width=1,
                             gain=1.0,
                             activation='relu',
                             bias=bias,
                             name=name+'-skip-1x1')
            
        if self.use_residual:            
            conv1_dilation = self.conv1d(outputs,
                             out_channels=self.num_hidden, # self.residual_channels,
                             filter_width=1,
                             gain=1.0,
                             activation='relu',
                             bias=bias,
                             name=name+'-res-1x1')
            # residual connection
            outputs = Add()([inputs, conv1_dilation])
        
        if self.use_skip:
            return skip, outputs
        
        return outputs
    
    def iterative_step_train(self, X, y, test_round=50, batch_size=20, epochs=50, test_step=10, y_feature_axis_in_X=0, should_norm_y=False, weight_file="wavenet_wt.h5"):
        
        to_predict = test_round * test_step
        ith_pos = X.shape[0] 
        preds = np.empty((0,))
        targets = np.empty((0,))
         
        print("Totally to predict: ", to_predict)
        print("Totally ", test_round, " iterative training/testing iterations")
        
        for i in range(ith_pos - to_predict, ith_pos, test_step):
            normalizer = Normalizer()
            end  = i+test_step  # to include ith_pos in last iteration
            X_ = X[:end, :]
            y_ = y[:end]    
            
            X_train, y_train, X_test, y_test  = split_batch_norm_NXNy(normalizer, test_step, X_, y_, y_feature_axis_in_X, should_norm_y, self.num_time_samples)
            
            if self.solution == 'classification':
                y_train = to_categorical(y_train, num_classes=self.num_classes)                
            else:
                y_train = y_train.astype(np.float32) 
            X_train = X_train.astype(np.float32)            
            X_test = X_test.astype(np.float32)
            print("X_train.shape: ", X_train.shape, "y_train.shape ", y_train.shape, self.use_condition, "self.num_channels ", self.num_channels)

            print("*******Iteration NO.: ", str(int((i-ith_pos+to_predict)/test_step)), ", number of training examples: ", X_train.shape[0], ", number of test example: ", X_test.shape[0], "*******")
            
            steps_per_epoch = int(np.ceil(X_train.shape[0] / float(batch_size)))
            print("batch_size: ", batch_size, "steps_per_epoch: ", steps_per_epoch)
            
            inp = Input(shape=(self.num_time_samples, self.num_channels ))
            out = self.create_network_loss(inp)
            model = Model(inputs=inp, outputs=out)
            if self.solution == 'classification':
                model.compile(optimizer = optimizers.Adam(lr=0.001),
                                   loss='categorical_crossentropy',
                                   metrics=['accuracy'])
            else:
                model.compile(optimizer = optimizers.Adam(lr=0.001),
                               loss='mean_absolute_error',
                               metrics=['accuracy'])
            
            model.fit_generator(generator(X_train, y_train, batch_size),
                  epochs=epochs,
                  # verbose=0, # stop print 
                  steps_per_epoch=steps_per_epoch) 
            # model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
            out = model.predict(X_test)
            out = np.argmax(out, axis=2)[:,-1] if self.solution == 'classification' else out[:, -1, :].reshape(-1)
            y_test = np.squeeze(y_test, axis=2)[:,-1]
            
            print("out.shape: ", out.shape, "out: ", out, "y_test.shape: ", y_test.shape, "y_test: ", y_test) 
            
            if should_norm_y: 
                out = normalizer.inverse_transform(out, y_feature_axis_in_X=y_feature_axis_in_X)
                y_test = normalizer.inverse_transform(y_test, y_feature_axis_in_X=y_feature_axis_in_X)

            preds = np.append(preds, out, axis=0)
            targets = np.append(targets, y_test, axis=0)
            
            # Save the model weights.  
            if (i-ith_pos+to_predict)==0 and weight_file is not None:
                self.weight_path = os.path.join('./', weight_file)
                model.save_weights(self.weight_path)
                print("path: ", self.weight_path)
            
            del model, X_train, y_train, X_test, y_test, out

        visualize_forecast_plot(preds, targets, show=False, save_figure=True, figname=weight_file.split('_wt')[0]+'_step_predict.eps')
        
        return targets, preds

    def iterative_train(self, X, y, test_round=600, batch_size=20, epochs=50, y_feature_axis_in_X=0, should_norm_y=False, weight_file="wavenet_wt.h5"):
                        
        iterations = X.shape[0] - self.num_time_samples - 1 # receptive_field == self.num_time_samples 
        preds = np.empty((0,))
        targets = np.empty((0,))
        
        print("Totally ", test_round, " iterative training/testing iterations")
        
        for i in range( max(0, iterations-test_round), iterations):
            normalizer = Normalizer()
            end  = self.num_time_samples + 2 + i # rf+1+i for train, 1 for test
            X_ = X[:end, :]
            y_ = y[:end]    
            
            X_train, y_train, X_test, y_test  = split_batch_norm_NXNy(normalizer, 1, X_, y_, y_feature_axis_in_X, should_norm_y, self.num_time_samples)
            
            if self.solution == 'classification':
                y_train = to_categorical(y_train, num_classes=self.num_classes)                
            else:
                y_train = y_train.astype(np.float32) 
            X_train = X_train.astype(np.float32)            
            X_test = X_test.astype(np.float32)
            print("X_train.shape: ", X_train.shape, "y_train.shape ", y_train.shape, "use_condition: ", self.use_condition, "self.num_channels ", self.num_channels)

            print("*******Iteration NO.: ", i-iterations+test_round, ", number of training examples: ", X_train.shape[0], ", number of test example: ", X_test.shape[0], "*******")

            steps_per_epoch = int(np.ceil(X_train.shape[0] / float(batch_size)))
            print("batch_size: ", batch_size, "steps_per_epoch: ", steps_per_epoch)
            
            inp = Input(shape=(self.num_time_samples, self.num_channels ))
            out = self.create_network_loss(inp)
            model = Model(inputs=inp, outputs=out)
            if self.solution == 'classification':
                model.compile(optimizer = optimizers.Adam(lr=0.001),
                                   loss='categorical_crossentropy',
                                   metrics=['accuracy'])
            else:
                model.compile(optimizer = optimizers.Adam(lr=0.001),
                               loss='mean_absolute_error',
                               metrics=['accuracy'])
            
            model.fit_generator(generator(X_train, y_train, batch_size),
                  epochs=epochs,
                  # verbose=0, # stop print 
                  steps_per_epoch=steps_per_epoch) 
            # model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
            out = model.predict(X_test)
            out = np.argmax(out, axis=2)[:,-1] if self.solution == 'classification' else out[:, -1, :].reshape(-1)
            y_test = np.squeeze(y_test, axis=2)[:,-1]
        
            print("out.shape: ", out.shape, "out: ", out, "y_test.shape: ", y_test.shape, "y_test: ", y_test)            
            preds = np.append(preds, out, axis=0)
            targets = np.append(targets, y_test, axis=0)
            
            # Save the model weights.  
            if ((i-iterations+test_round == 0) or i==0) and weight_file is not None:
                # self.generate(X_train[-1,-1,0][None,None], X_train[-1,-1,1][None,None], test_round, y_test)
                self.weight_path = os.path.join('./', weight_file)
                model.save_weights(self.weight_path)
                print("path: ", self.weight_path) 
            
        visualize_forecast_plot(preds, targets, show=False, save_figure=True, figname=weight_file.split('_wt')[0]+'_predict.png')
        
        return targets, preds
        
    def train(self, X_train, y_train, batch_size, epochs, steps_per_epoch, weight_file='wt.h5'):
        
        X_train = X_train.astype(np.float32)
       
        if self.solution == 'classification':
            y_train = tf.squeeze(tf.one_hot(y_train, self.num_classes), axis=2) 
        else:
            y_train = y_train.astype(np.float32)        
          
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        X_batch, y_batch = iterator.get_next() 
        
        model_input = Input(tensor=X_batch)
        model_output = self.create_network_loss(model_input)
        self.model = Model(inputs = model_input,  outputs = model_output)

        # multiple output
        # model_input_X = Input(tensor=X_batch)
        # model_input_C = Input(tensor=C_batch)                  
        # model_output = self.create_network_loss(model_input_X, model_input_C)
        # self.model = Model(inputs = [model_input_X, model_input_C],  outputs = model_output)
        
        print("model_input.shape: ", model_input.shape, "model_output.shape: ", model_output.shape, "use_condition: ", self.use_condition)
        
        if self.solution == 'classification':
            self.model.compile(optimizer = optimizers.Adam(lr=0.001),
                        loss = 'categorical_crossentropy', 
                        metrics = ['accuracy'],
                        target_tensors = [y_batch]
                         )
            if weight_file is not None:
                weight_file = 'c_' + weight_file
        else:
            self.model.compile(optimizer = optimizers.Adam(lr=0.001),
                        loss = 'mean_absolute_error', 
                        metrics = ['accuracy'],
                        target_tensors = [y_batch]
                         )
            if weight_file is not None:
                weight_file = 'r_' + weight_file
        
        self.model.fit(x=None, y=None, 
                  epochs=epochs,
                  # verbose=0, # stop print 
                  steps_per_epoch=steps_per_epoch)
          
        # Save the model weights.  
        if weight_file is not None:
            self.weight_path = os.path.join('./', weight_file)
            self.model.save_weights(self.weight_path)
            print("path: ", self.weight_path)        
        
        return self.model
    
    def _load_model(self, weight_path):

        # Second session to test loading trained model without tensors.   
        inp = Input(shape=(self.num_time_samples, self.num_channels ))
        out = self.create_network_loss(inp)
        model = Model(inputs=inp, outputs=out)
        
        model.load_weights(weight_path)
        
        if self.solution == 'classification':
            model.compile(optimizer = optimizers.Adam(lr=0.001),
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        else:
            model.compile(optimizer = optimizers.Adam(lr=0.001),
                           loss='mean_absolute_error',
                           metrics=['accuracy'])
        return model
        
    def predict(self, X_test, y_test, plot=True, weight_path=None):

        if weight_path is not None:
            # Clean up the TF session.
            # K.clear_session()
            test_model = self._load_model(weight_path)      
        else:
            test_model = self._load_model(self.weight_path)

        predictions = test_model.predict(X_test)        
        pred_outs = np.argmax(predictions, axis=2)[:,-1] if self.solution == 'classification' else predictions[:, -1, :].reshape(-1)
        y_test = np.squeeze(y_test, axis=2)[:,-1]
 
        if plot:
            visualize_forecast_plot(pred_outs, y_test)
  
        return pred_outs, y_test
    
    def _causal_linear(self, model, inputs, state, name, bias=True, activation=None):
    
        w = model.get_layer(name).get_weights()[0]

        # filter_width = 2
        w_r = w[0, :, :]
        w_e = w[1, :, :]

        if bias:
            b = model.get_layer(name).get_weights()[1]
            output = tf.matmul(inputs, w_e) + tf.matmul(state, w_r) + tf.expand_dims(b, 0)
        else:
            output = tf.matmul(inputs, w_e) + tf.matmul(state, w_r)

        if activation:
            output = Activation(activation)(output)
      
        return output

    def _output_linear(self, model, h, name, bias=True, activation=None):

        w = model.get_layer(name).get_weights()[0]
        
        # filter_width = 1
        w = w[0, :, :]
        
        if bias:            
            b = model.get_layer(name).get_weights()[1]
            output = tf.matmul(h, w) + tf.expand_dims(b, 0)
        else:
            output = tf.matmul(h, w)

        if activation:
            output = Activation(activation)(output)
            
        return output

    def _block_output(self, model, inputs, state, condition, name, last_layer=False, bias=True, activation=None):
        
        conv_filter = self._causal_linear(model, inputs, state, name+'-filter-x', bias, activation=None)       
        conv_gate = self._causal_linear(model, inputs, state, name+'-gate-x', bias, activation=None)
        
        if condition is not None and self.use_condition:
            conv_filter = conv_filter + self._output_linear(model, condition, name+'-filter-c', bias, activation=None)
            conv_gate = conv_gate + self._output_linear(model, condition, name+'-gate-c', bias, activation=None)
        
        output = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)  
        
        if self.use_skip:
            skip = self._output_linear(model, output, name+'-skip-1x1', bias, activation)
          
        if self.use_residual and not last_layer:
            res = self._output_linear(model, output, name+'-res-1x1', bias, activation)
            output = res + inputs
      
        if self.use_skip:
            return skip, output
          
        return output
      
    def _init_generator(self, model, batch_size=1, input_size=1):
        
        input_ = tf.placeholder(tf.float32, [batch_size, input_size],
                                name='inputs')
        condition_ = tf.placeholder(tf.float32, [batch_size, input_size],
                                name='inputs')
#         if input_.shape[-1] > 1 and self.use_condition:
#             h = input_[:,0][None]
#             condition_ = input_[:,1][None]
#         else:
#             h = input_[:,0][None]
#             condition_ = None
        count = 0
        h = input_
        init_ops = []
        push_ops = []
        skips = []
        
        h = self._output_linear(model, h, 'ipt-1x1', activation='relu')
        
        for b in range(self.num_blocks):
            for i in range(self.num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                #if count == 0:
                #    state_size = 1
                #else:
                # ipt-1x1 increased channels
                state_size = self.num_hidden # self.residual_channels
                    
                q = tf.FIFOQueue(rate,
                                 dtypes=tf.float32,
                                 shapes=(batch_size, state_size))
                init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))    
                
                state_ = q.dequeue()
                push = q.enqueue([h])
                
                init_ops.append(init)
                push_ops.append(push)
                
                if self.use_skip:                    
                    if count == self.num_blocks * self.num_layers - 1:
                        skip, h = self._block_output(model, h, state_, condition_, name=name, last_layer=True, activation='relu')
                    else:
                        skip, h = self._block_output(model, h, state_, condition_, name=name, activation='relu')
                    skips.append(skip)
                else:
                    h = self._block_output(model, h, state_, condition_, name=name, activation='relu')
                count += 1
        
        # final outputs
        if self.solution == 'classification':
            if self.use_skip:
                out = Activation('relu')(Add()(skips))
                out_ops = [self._output_linear(model, out, 'f-1x1', activation='softmax')]
            else:
                out_ops = [self._output_linear(model, h, 'f-1x1', activation='softmax')]
        else:
            if self.use_skip:
                out = Activation('relu')(Add()(skips))
                out_ops = [self._output_linear(model, out, 'f-1x1', activation='linear')]
            else:
                out_ops = [self._output_linear(model, h, 'f-1x1', activation='linear')]
        # put together with other updated recurrent states
        out_ops.extend(push_ops) 
        
        #self.inputs = [input_, condition_]
        self.inputs = input_
        self.conditions = condition_
        self.init_ops = init_ops
        self.out_ops = out_ops
        
        # Initialize queues.
        self.sess = K.get_session()
        self.sess.run(self.init_ops)

    def generate(self, normalizer, bins, input_, condition_, num_samples, y_test, weight_path, y_feature_axis_in_X=0, cond_feature_axis_in_X=1, batch_size=1, input_size=1):

        gen_model = self._load_model(weight_path) 
            
        self._init_generator(gen_model, batch_size=batch_size, input_size=input_.shape[-1])
        # self._init_generator(gen_model, batch_size, input_size)
        
        # keep disretisation or unnormed regression value
        predictions = []
        condition_ = normalizer.transform(condition_, y_feature_axis_in_X=cond_feature_axis_in_X)
        input_ = normalizer.transform(input_, y_feature_axis_in_X=y_feature_axis_in_X) 

        for step in range(num_samples):           
            #feed_dict = {self.inputs: [input_, condition]}
            feed_dict = {self.inputs: input_, self.conditions: condition_}
            
            # ignore push ops, only keep final output, iteratively call q.enqueue(h) and compute h in all related layers
            output = self.sess.run(self.out_ops, feed_dict=feed_dict)[0] 
            
            if self.solution == 'classification':
                # dim of output = (batch_size, num_classes)
                # quence shifts to left one-step 
                value = np.argmax(output[0, :], axis = -1)
                # arr[None, None] = arr[None, None, :], add new axis to reshap to (1,1,len(arr))
                # decode index of bin, and rename input_ to feed back to model
                input_ = np.array(bins[value])[None, None] # (1,1)      
                # norm bins[value], otherwise input_ is already normed, no need for further norm
                input_ = normalizer.transform(input_, y_feature_axis_in_X=y_feature_axis_in_X) 
            else:
                value = output[0, :]
                input_ = np.array([value])
                        
            # self.use_condition = False
            
            # input_ will always be normed value for both cases: regression and classification
            print("step ", step, " prediction: ", input_, " target: ", y_test[step])
            predictions.append(value)
    
        predictions_ = np.concatenate(predictions, axis=0).reshape(-1)
        # invser_norm for regression, otherwise, keep disretisation
        if self.solution != 'classification':
            predictions_ = normalizer.inverse_transform(predictions_, y_feature_axis_in_X=y_feature_axis_in_X) 
        visualize_forecast_plot(predictions_, y_test, show=False, save_figure=True, figname=weight_path.split('_wt')[0]+'_generate.eps')
        
        return predictions_
        
        
        

        