#-*-coding:utf-8-*-
import os
import re
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs
import cPickle

from utils import shared, set_values, get_name
from nn import HiddenLayer, EmbeddingLayer, DropoutLayer, LSTM, GRU, forward, LSTM_d
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
            # model_path = os.path.join(models_path, self.name)
            # self.model_path = model_path
	    model_path = '1'
            self.model_path = model_path
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

    def save_mappings(self, id_to_word, id_to_char, id_to_tag):#, id_to_pos):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        # self.id_to_pos = id_to_pos
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_char': self.id_to_char,
                'id_to_tag': self.id_to_tag,
                # 'id_to_pos': self.id_to_pos,
            }
            cPickle.dump(mappings, f)

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = cPickle.load(f)
        self.id_to_word = mappings['id_to_word']
        self.id_to_char = mappings['id_to_char']
        self.id_to_tag = mappings['id_to_tag']
        # self.id_to_pos = mappings['id_to_pos']

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
              word_dim,
              word_hidden_dim,
              word_bidirect,
              tagger_hidden_dim,
              hamming_cost,
              L2_reg,
              lr_method,
              pre_word_emb,
              pre_char_emb,
              tagger,
              use_gaze,
              POS,
              plot_cost,
              #cap_dim,
              training=True,
              **kwargs
              ):
        """
        Build the network.
        """
        # Training parameters
        n_words = len(self.id_to_word)
        n_chars = len(self.id_to_char)
        n_tags = len(self.id_to_tag)
        # n_pos = len(self.id_to_pos) + 1

        # Number of capitalization features
        #if cap_dim:
        #    n_cap = 4

        # Network variables
        is_train = T.iscalar('is_train') # declare variable,声明整型变量is_train
        word_ids = T.ivector(name='word_ids') #声明整型一维向量
        char_for_ids = T.imatrix(name='char_for_ids') # 声明整型二维矩阵
        char_rev_ids = T.imatrix(name='char_rev_ids')
        char_pos_ids = T.ivector(name='char_pos_ids')
        if use_gaze:
            gaze = T.imatrix(name='gaze')
        if POS:
            # pos_ids = T.ivector(name='pos_ids')
            pos_one_hot = T.imatrix(name= 'pos_one_hot')
        #hamming_cost = T.matrix('hamming_cost', theano.config.floatX) # 声明整型二维矩阵
        tag_ids = T.ivector(name='tag_ids')
        #if cap_dim:
        #    cap_ids = T.ivector(name='cap_ids')

        # Sentence length
        s_len = (word_ids if word_dim else char_pos_ids).shape[0]  #句子中的单词数

        # Final input (all word features)
        input_dim = 0
        inputs = []
        L2_norm = 0.0

        theano.config.compute_test_value = 'off'
        #
        # Word inputs
        #
        if word_dim:
            input_dim += word_dim
            word_layer = EmbeddingLayer(n_words, word_dim, name='word_layer')
            word_input = word_layer.link(word_ids)
            inputs.append(word_input)
            # Initialize with pretrained embeddings
            if pre_word_emb and training:
                new_weights = word_layer.embeddings.get_value()
                print 'Loading pretrained word embeddings from %s...' % pre_word_emb
                pretrained = {}
                emb_invalid = 0
                for i, line in enumerate(codecs.open(pre_word_emb, 'r', 'utf-8', 'ignore')):
                    line = line.rstrip().split()
                    if len(line) == word_dim + 1:
                        pretrained[line[0]] = np.array(
                            [float(x) for x in line[1:]]
                        ).astype(np.float32)
                    else:
                        emb_invalid += 1
                if emb_invalid > 0:
                    print 'WARNING: %i invalid word embedding lines' % emb_invalid
                c_found = 0
                c_lower = 0
                c_zeros = 0
                # Lookup table initialization
                for i in xrange(n_words):
                    word = self.id_to_word[i]
                    if word in pretrained:
                        new_weights[i] = pretrained[word]
                        c_found += 1
                    elif word.lower() in pretrained:
                        new_weights[i] = pretrained[word.lower()]
                        c_lower += 1
                    elif re.sub('\d', '0', word) in pretrained:
                        new_weights[i] = pretrained[
                            re.sub('\d', '0', word)
                        ]
                        c_zeros += 1
                word_layer.embeddings.set_value(new_weights)
                print 'Loaded %i pretrained word embeddings.' % len(pretrained)
                print ('%i / %i (%.4f%%) words have been initialized with '
                       'pretrained word embeddings.') % (
                            c_found + c_lower + c_zeros, n_words,
                            100. * (c_found + c_lower + c_zeros) / n_words
                      )
                print ('%i found directly, %i after lowercasing + zero.') % (c_found, c_lower + c_zeros)
            L2_norm += (word_layer.embeddings ** 2).sum()


        #
        # Chars inputs
        #
        if char_dim:
            input_dim += char_hidden_dim
            char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')
            char_for_input = char_layer.link(char_for_ids)
            char_rev_input = char_layer.link(char_rev_ids)

            # Initialize with pretrained char embeddings
            if pre_char_emb and training:
                new_weights = char_layer.embeddings.get_value()
                print 'Loading pretrained char embeddings from %s...' % pre_char_emb
                pretrained = {}
                emb_invalid = 0
                for i, line in enumerate(codecs.open(pre_char_emb, 'r', 'utf-8', 'ignore')):
                    line = line.rstrip().split()
                    if len(line) == char_dim + 1:
                        pretrained[line[0]] = np.array(
                            [float(x) for x in line[1:]]
                        ).astype(np.float32)
                    else:
                        emb_invalid += 1
                if emb_invalid > 0:
                    print 'WARNING: %i invalid char embedding lines' % emb_invalid
                c_found = 0
                c_lower = 0
                c_zeros = 0
                # Lookup table initialization
                for i in xrange(n_chars):
                    char = self.id_to_char[i]
                    if char in pretrained:
                        new_weights[i] = pretrained[char]
                        c_found += 1
                    elif word.lower() in pretrained:
                        new_weights[i] = pretrained[word.lower()]
                        c_lower += 1
                    elif re.sub('\d', '0', char) in pretrained:
                        new_weights[i] = pretrained[
                            re.sub('\d', '0', char)
                        ]
                        c_zeros += 1
                char_layer.embeddings.set_value(new_weights)
                print 'Loaded %i pretrained char embeddings.' % len(pretrained)
                print ('%i / %i (%.4f%%) words have been initialized with '
                       'pretrained char embeddings.') % (
                            c_found + c_lower + c_zeros, n_chars,
                            100. * (c_found + +c_lower + c_zeros) / n_chars
                      )
                print ('%i found directly, %i after lowercasing + zero.') % (c_found, c_lower + c_zeros)
            L2_norm += (char_layer.embeddings ** 2).sum()


	    char_lstm_for = LSTM(char_dim, char_hidden_dim, with_batch=True, name='char_lstm_for')
            char_lstm_rev = LSTM(char_dim, char_hidden_dim, with_batch=True, name='char_lstm_rev')

            char_lstm_for.link(char_for_input)
            char_lstm_rev.link(char_rev_input)

            char_for_output = char_lstm_for.h.dimshuffle((1, 0, 2))[T.arange(s_len), char_pos_ids]
            char_rev_output = char_lstm_rev.h.dimshuffle((1, 0, 2))[T.arange(s_len), char_pos_ids]

            for param in char_lstm_for.params[:8]:
                L2_norm += (param ** 2).sum()

	    if char_bidirect:
		char_lstm_hidden = T.concatenate([char_for_output, char_rev_output], axis= 1)
                input_dim += char_hidden_dim
                for param in char_lstm_rev.params[:8]:
                    L2_norm += (param ** 2).sum()

            else:
	        char_lstm_hidden = char_for_output


            inputs.append(char_lstm_hidden)

        # if POS:
            # pos_dim = 20
            # input_dim += pos_dim
            # pos_layer = EmbeddingLayer(n_pos, pos_dim, name='pos_layer')
            # pos_input = pos_layer.link(pos_ids)
            # inputs.append(pos_input)
            # L2_norm += (pos_layer.embeddings ** 2).sum()


        #if len(inputs) != 1:
        inputs = T.concatenate(inputs, axis=1)

        #
        # Dropout on final input
        #
        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(inputs)
            input_test = (1 - dropout) * inputs
            inputs = T.switch(T.neq(is_train, 0), input_train, input_test) # 条件句

        # if POS:
        #     inputs = T.concatenate([inputs, pos_one_hot], axis= 1)
        #     input_dim += 6

        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_hidden_dim, with_batch=False,
                             name='word_lstm_for')
        word_lstm_rev = LSTM(input_dim, word_hidden_dim, with_batch=False,
                             name='word_lstm_rev')
        word_lstm_for.link(inputs)          # 单词的顺序： I like dog
        word_lstm_rev.link(inputs[::-1, :]) # 单词的顺序： dog like I
        word_for_output = word_lstm_for.h
        word_rev_output = word_lstm_rev.h[::-1, :]

        for param in word_lstm_for.params[:8]:
            L2_norm += (param ** 2).sum()

        if word_bidirect:
            final_output = T.concatenate(
                [word_for_output, word_rev_output],
                axis=1
            )

            tanh_layer = HiddenLayer(2 * word_hidden_dim, word_hidden_dim,
                                     name='tanh_layer', activation='tanh')
            final_output = tanh_layer.link(final_output)
            for param in word_lstm_rev.params[:8]:
                L2_norm += (param ** 2).sum()

        else:
            final_output = word_for_output


        dims = word_hidden_dim
        if use_gaze:
            final_output = T.concatenate([final_output, gaze], axis= 1)
            dims = word_hidden_dim + n_tags


        if POS:
            final_output = T.concatenate([final_output, pos_one_hot], axis=1)
            dims += 6




        # if word_bidirect:
        #     final_output = T.concatenate(
        #         [word_for_output, word_rev_output],
        #         axis=1
        #     )
        #     tanh_layer = HiddenLayer(2 * word_hidden_dim, word_hidden_dim,
        #                              name='tanh_layer', activation='tanh')
        #     final_output = tanh_layer.link(final_output)
        # else:
        #     final_output = word_for_output

        # Sentence to Named Entity tags
        ## final_layer = HiddenLayer(dims, n_tags, name='final_layer',
        ##                           activation=(None if crf else 'softmax'))
        # final_layer = HiddenLayer(word_hidden_dim, n_tags, name='final_layer',
        #                           activation=(None if crf else 'softmax'))
        ## tags_scores = final_layer.link(final_output)
        ## L2_norm += (final_layer.params[0] ** 2).sum()

        # No CRF
        if tagger == 'lstm':
            tagger_layer = LSTM_d(dims, tagger_hidden_dim, with_batch= False, name='LSTM_d')
            tagger_layer.link(final_output)
            final_output = tagger_layer.t

            dims = tagger_hidden_dim

            for param in tagger_layer.params[:8]:
                L2_norm += (param ** 2).sum()

        final_layer = HiddenLayer(dims, n_tags, name='final_layer',
                                  activation=(None if tagger == 'crf' else 'softmax'))
        tags_scores = final_layer.link(final_output)
        L2_norm += (final_layer.params[0] ** 2).sum()

        if tagger != 'crf':
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
            all_paths_scores = forward(observations, transitions, hamming_cost=hamming_cost, n_tags=n_tags, padded_tags_ids=padded_tags_ids)
            L2_norm += (transitions ** 2).sum()
            cost = - (real_path_score - all_paths_scores) + L2_reg * L2_norm

        # Network parameters
        params = []
        if word_dim:
            self.add_component(word_layer)
            params.extend(word_layer.params)
        if char_dim:
            self.add_component(char_layer)
            params.extend(char_layer.params)
            self.add_component(char_lstm_for)
            params.extend(char_lstm_for.params)

            if char_bidirect:
                self.add_component(char_lstm_rev)
            	params.extend(char_lstm_rev.params)

        self.add_component(word_lstm_for)
        params.extend(word_lstm_for.params)

        # if POS:
        #     self.add_component(pos_layer)
        #     params.extend(pos_layer.params)

        if word_bidirect:
            self.add_component(word_lstm_rev)
            params.extend(word_lstm_rev.params)


        self.add_component(final_layer)
        params.extend(final_layer.params)

        if tagger == 'lstm':
            self.add_component(tagger_layer)
            params.extend(tagger_layer.params)
        elif tagger == 'crf':
            self.add_component(transitions)
            params.append(transitions)

        if word_bidirect:
            self.add_component(tanh_layer)
            params.extend(tanh_layer.params)



        # Prepare train and eval inputs
        eval_inputs = []
        if word_dim:
            eval_inputs.append(word_ids)
        if use_gaze:
            eval_inputs.append(gaze)
        if POS:
            # eval_inputs.append(pos_ids)
            eval_inputs.append(pos_one_hot)
        if char_dim:
            eval_inputs.append(char_for_ids)
            if char_bidirect:
                eval_inputs.append(char_rev_ids)
            eval_inputs.append(char_pos_ids)
        #if cap_dim:
        #    eval_inputs.append(cap_ids)
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
                givens=({is_train: np.cast['int32'](1)} if dropout else {}),
                on_unused_input='warn'
            )
        else:
            f_train = None

        if plot_cost:
            f_plot_cost = theano.function(
                inputs=train_inputs,
                outputs=cost,
                givens=({is_train: np.cast['int32'](1)} if dropout else {}),
                on_unused_input='warn'
            )
        else:
            f_plot_cost = None

        # Compile evaluation function
        if tagger != 'crf':
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores,
                givens=({is_train: np.cast['int32'](0)} if dropout else {}),
                on_unused_input='warn'
            )
        else:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=forward(observations, transitions, hamming_cost= 0, n_tags= None, padded_tags_ids= None, viterbi=True,
                                return_alpha=False, return_best_sequence=True),
                givens=({is_train: np.cast['int32'](0)} if dropout else {}),
                on_unused_input='warn'
            )

        return f_train, f_eval, f_plot_cost
