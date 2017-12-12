#-*-coding:utf-8-*-
import os
import re
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs
import cPickle

from utils import shared, set_values, get_name, FOFE
from nn import HiddenLayer, EmbeddingLayer, DropoutLayer, LSTM, GRU, forward
from optimization import Optimization


class Model(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, models_path=None, model_path=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        if model_path is None:
            assert parameters and models_path
            # Create a name based on the parameters
            self.parameters = parameters
            self.name = get_name(parameters)
            # Model location
            model_path = os.path.join(models_path, self.name)
            self.model_path = model_path
	    # model_path = os.path.join(models_path， '1'）
         #    self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                cPickle.dump(parameters, f)
        else:
            assert parameters is None and models_path is None
            # Model location
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Load the parameters and the mappings from disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters = cPickle.load(f)
            self.reload_mappings()
        self.components = {}

    def save_mappings(self, id_to_char, id_to_tag):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_char': self.id_to_char,
                'id_to_tag': self.id_to_tag,
            }
            cPickle.dump(mappings, f)

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = cPickle.load(f)
        self.id_to_char = mappings['id_to_char']
        self.id_to_tag = mappings['id_to_tag']

    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def save(self):
        """
        Write components values to disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)

    def reload(self):
        """
        Load components values from disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])


    def build(self,
              dropout,
              char_dim,
              char_hidden_dim,
              char_bidirect,
              layer2_hidden_dim,
              lr_method,
              layer2,
              batch_size,
              pre_emb,
              use_gaze,
              crf,
              training=True,
              **kwargs
              ):
        """
        Build the network.
        """
        # Training parameters
        n_chars = len(self.id_to_char)
        n_tags = len(self.id_to_tag)


        # Network variables
        is_train = T.iscalar('is_train') # declare variable,声明整型变量is_train
        char_ids = T.ivector(name='char_ids') #声明整型一维向量
        if use_gaze:
            gaze = T.imatrix(name='gaze')
        #hamming_cost = T.matrix('hamming_cost', theano.config.floatX) # 声明整型二维矩阵
        # tag_ids = T.imatrix(name='tag_ids')
        tag_ids = T.ivector(name='tag_ids')
        # Sentence length
        s_len = char_ids.shape[0]  #每个句子中的字数

        # Final input (all word features)
        #
        # Char inputs
        #
        if char_dim:
            char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')
            char_input = char_layer.link(char_ids)
            # Initialize with pretrained embeddings
            if pre_emb and training:
                new_weights = char_layer.embeddings.get_value()
                print 'Loading pretrained embeddings from %s...' % pre_emb
                pretrained = {}
                emb_invalid = 0
                for i, line in enumerate(codecs.open(pre_emb, 'r', 'utf-8', 'ignore')):
                    line = line.rstrip().split()
                    if len(line) == char_dim + 1:
                        pretrained[line[0]] = np.array(
                            [float(x) for x in line[1:]]
                        ).astype(np.float32)
                    else:
                        emb_invalid += 1
                if emb_invalid > 0:
                    print 'WARNING: %i invalid lines' % emb_invalid
                c_found = 0
                c_lower = 0
                c_zeros = 0
                # Lookup table initialization
                for i in xrange(n_chars):
                    char = self.id_to_char[i]
                    if char in pretrained:
                        new_weights[i] = pretrained[char]
                        c_found += 1
                    elif char.lower() in pretrained:
                        new_weights[i] = pretrained[char.lower()]
                        c_lower += 1
                    elif re.sub('\d', '0', char) in pretrained:
                        new_weights[i] = pretrained[
                            re.sub('\d', '0', char)
                        ]
                        c_zeros += 1
                char_layer.embeddings.set_value(new_weights)
                print 'Loaded %i pretrained embeddings.' % len(pretrained)
                print ('%i / %i (%.4f%%) chars have been initialized with '
                       'pretrained embeddings.') % (
                            c_found + c_lower + c_zeros, n_chars,
                            100. * (c_found + c_lower + c_zeros) / n_chars
                      )
                print ('%i found directly, %i after lower, %i after zero.') % (c_found, c_lower, c_zeros)


        #
        # Dropout on final input
        #
        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(char_input)
            input_test = (1 - dropout) * char_input
            char_input = T.switch(T.neq(is_train, 0), input_train, input_test) # 条件句




        # LSTM for chars, first layer
        char_lstm_for1 = LSTM(char_dim, char_hidden_dim, with_batch= False,
                                 name='first_char_lstm_for')
        char_lstm_rev1 = LSTM(char_dim, char_hidden_dim, with_batch= False,
                                 name='first_char_lstm_rev')
        char_lstm_for1.link(char_input)          # char的顺序： l i k e
        char_lstm_rev1.link(char_input[::-1, :]) # 单词的顺序： e k i l
        char_for_output1 = char_lstm_for1.h
        char_rev_output1 = char_lstm_rev1.h[::-1, :]


        if char_bidirect:
            final_output = T.concatenate(
                        [char_for_output1, char_rev_output1],
                        axis=1
            )
            tanh_layer1 = HiddenLayer(2 * char_hidden_dim, char_hidden_dim,
                                             name='tanh_layer1', activation='tanh')
            final_output = tanh_layer1.link(final_output)
        else:
            final_output = char_for_output1

        
        if layer2:
            #
            # Dropout on final input
            #
            if dropout:
                dropout_layer = DropoutLayer(p=dropout)
                input_train = dropout_layer.link(final_output)
                input_test = (1 - dropout) * final_output
                final_output = T.switch(T.neq(is_train, 0), input_train, input_test)  # 条件句

            # LSTM for chars, second layer
            char_lstm_for2 = LSTM(char_hidden_dim, layer2_hidden_dim, with_batch= False,
                                 name='second_char_lstm_for')
            char_lstm_rev2 = LSTM(char_hidden_dim, layer2_hidden_dim, with_batch= False,
                                 name='second_char_lstm_rev')
            char_lstm_for2.link(final_output)
            char_lstm_rev2.link(final_output[::-1, :])
            char_for_output2 = char_lstm_for2.h
            char_rev_output2 = char_lstm_rev2.h[::-1, :]


            if char_bidirect:
                final_output = T.concatenate(
                            [char_for_output2, char_rev_output2],
                            axis=1
                )
                tanh_layer2 = HiddenLayer(2 * layer2_hidden_dim, layer2_hidden_dim,
                                                 name='tanh_layer2', activation='tanh')
                final_output = tanh_layer2.link(final_output)
            else:
                final_output = char_for_output2


        if layer2:
            dims = layer2_hidden_dim
        else:
            dims = char_hidden_dim

        if use_gaze:
            final_output = T.concatenate([final_output, gaze], axis= 1)
            dims = dims + n_tags

        # final_output = T.reshape(final_output, (-1, input_dim))

        # Sentence to Named Entity tags - Score，ci与CRF之间的隐含层
        final_layer = HiddenLayer(dims, n_tags, name='final_layer',
                                  activation=(None if crf else 'softmax'))
        tags_scores = final_layer.link(final_output)

        # No CRF
        if not crf:
            cost = T.nnet.categorical_crossentropy(tags_scores, tag_ids).mean()
        # CRF
        else:
            transitions = shared((n_tags + 2, n_tags + 2), 'transitions')

            small = -1000
            b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
            e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)
            observations = T.concatenate(
                [tags_scores, small * T.ones((s_len, 2))],
                axis=1
            )
            observations = T.concatenate(
                [b_s, observations, e_s],
                axis=0
            )

            # Score from tags
            real_path_score = tags_scores[T.arange(s_len), tag_ids].sum() # P中对应元素的求和好

            # Score from add_componentnsitions
            b_id = theano.shared(value=np.array([n_tags], dtype=np.int32))
            e_id = theano.shared(value=np.array([n_tags + 1], dtype=np.int32))
            padded_tags_ids = T.concatenate([b_id, tag_ids, e_id], axis=0)
            real_path_score += transitions[
                padded_tags_ids[T.arange(s_len + 1)],
                padded_tags_ids[T.arange(s_len + 1) + 1]
            ].sum()  # A中对应元素的求和
	   
            
            all_paths_scores = forward(observations, transitions) 
	    cost = - (real_path_score - all_paths_scores)

        # Network parameters
        params = []
        if char_dim:
            self.add_component(char_layer)
            params.extend(char_layer.params)

        self.add_component(char_lstm_for1)
        params.extend(char_lstm_for1.params)
        if char_bidirect:
            self.add_component(char_lstm_rev1)
            params.extend(char_lstm_rev1.params)

            self.add_component(tanh_layer1)
            params.extend(tanh_layer1.params)

        if layer2:
            self.add_component(char_lstm_for2)
            params.extend(char_lstm_for2.params)
            if char_bidirect:
                self.add_component(char_lstm_rev2)
                params.extend(char_lstm_rev2.params)

                self.add_component(tanh_layer2)
                params.extend(tanh_layer2.params)

        self.add_component(final_layer)
        params.extend(final_layer.params)
        if crf:
            self.add_component(transitions)
            params.append(transitions)


        # Prepare train and eval inputs
        eval_inputs = []
        if char_dim:
            eval_inputs.append(char_ids)
        if use_gaze:
            eval_inputs.append(gaze)
        train_inputs = eval_inputs + [tag_ids]

        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Compile training function
        print 'Compiling...'
        if training:
            updates = Optimization(clip=5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters)
            f_train = theano.function(
                inputs=train_inputs,
                outputs=cost,
                updates=updates,
                givens=({is_train: np.cast['int32'](1)} if dropout else {})
            )
        else:
            f_train = None

        # Compile evaluation function
        if not crf:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores,
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
        else:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=forward(observations, transitions, viterbi=True,
                                return_alpha=False, return_best_sequence=True),
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )

        return f_train, f_eval
