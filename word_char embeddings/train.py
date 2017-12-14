#!/usr/bin/env python
#-*-coding:utf-8-*-

import os
import numpy as np
import optparse
import itertools
from collections import OrderedDict
from utils import create_input
import loader
import codecs
from utils import models_path, evaluate, eval_temp
from loader import word_mapping, char_mapping, tag_mapping, pos_mapping, read_list
from loader import update_tag_scheme, prepare_dataset
from loader import augment_with_pretrained
from model import Model
import time

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB,IOESB or IOES,B1,B2)"
)
optparser.add_option(
   "-l", "--lower", default="1",
   type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-O", "--POS", default="0",
    type='int', help="Use part-of-speech"
)
optparser.add_option(
    "-U", "--use_gaze", default="0",
    type='int', help="Use gaze"
)
optparser.add_option(
    "-S", "--post_process", default="0",
    type='int', help="Use post-process"
)
optparser.add_option(
    "-g", "--dictionary", default="",
    help="Gaze dictionary"
)
optparser.add_option(
    "-c", "--char_dim", default="50",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-H", "--hamming_cost", default="0",
    type='int', help="hamming cost"
)
optparser.add_option(
    "-R", "--L2_reg", default="0.0",
    type='float', help="hamming cost"
)
optparser.add_option(
    "-C", "--char_hidden_dim", default="50",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM/GRU for chars"
)

optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_hidden_dim", default="100",
    type='int', help="Token LSTM/GRU hidden layer size"
)
optparser.add_option(
    "-a", "--tagger_hidden_dim", default="20",
    type='int', help="tagger embedding dimension"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)

optparser.add_option(
    "-p", "--pre_word_emb", default="",
    help="Location of pretrained word embeddings"
)
optparser.add_option(
    "-P", "--pre_char_emb", default="",
    help="Location of pretrained char embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="0",
    type='int', help="Load all embeddings"
)
#optparser.add_option(
#    "-a", "--cap_dim", default="0",
#    type='int', help="Capitalization feature dimension (0 to disable)"
#)
optparser.add_option(
    "-f", "--tagger", default="crf",
    help="Use CRF/Softmax/LSTM_d (0 to disable)"
)
optparser.add_option(
    "-o", "--plot_cost", default="1",
    type='int', help="Plot cost line (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.005",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
opts = optparser.parse_args()[0]

# Parse parameters
parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['POS'] = opts.POS == 1
parameters['use_gaze'] = opts.use_gaze == 1
parameters['plot_cost'] = opts.plot_cost == 1
parameters['hamming_cost'] = opts.hamming_cost
parameters['L2_reg'] = opts.L2_reg
parameters['char_dim'] = opts.char_dim
parameters['char_hidden_dim'] = opts.char_hidden_dim
parameters['tagger_hidden_dim'] = opts.tagger_hidden_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_hidden_dim'] = opts.word_hidden_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_word_emb'] = opts.pre_word_emb
parameters['pre_char_emb'] = opts.pre_char_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['dictionary'] = opts.dictionary
#parameters['cap_dim'] = opts.cap_dim
parameters['tagger'] = opts.tagger
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method

# Check parameters validity
# print "trainfile:", opts.train
# print "testfile:", opts.test
# print "devfile:", opts.dev
# print os.path.abspath(os.curdir)
assert os.path.isfile(opts.train)
#assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes', 'ioesb1b2']
assert not parameters['all_emb'] or parameters['pre_word_emb']
assert not parameters['pre_word_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_word_emb'] or os.path.isfile(parameters['pre_word_emb'])

# Check evaluation script / folders
#if not os.path.isfile(eval_script):
#    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
#if not os.path.exists(eval_temp):
#    os.makedirs(eval_temp)
#if not os.path.exists(models_path):
#    os.makedirs(models_path)

# Initialize model
start_time = time.time() 
model = Model(parameters=parameters, models_path=models_path)
print "Trained model location: %s" % (model.model_path)
#, reload model location: %s" % (model.models_path, model.model_path)

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
pos = parameters['POS']
use_gaze = parameters['use_gaze']
tag_scheme = parameters['tag_scheme']
post_process = opts.post_process == 1
# Load sentences
train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
test_sentences = loader.load_sentences(opts.test,lower, zeros)

# Use selected tagging scheme (IOB / IOESB / IOESB1B2)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_word_emb']:
    dico_words_train = word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_word_emb'],
        list(itertools.chain.from_iterable([[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    dico_words_train = dico_words

# Create a dictionary and a mapping for words / POS tags / tags
# If we use pretrained embeddings, we add them to the dictionary.

if parameters['pre_char_emb']:
    dico_chars_train = char_mapping(train_sentences)[0]
    dico_chars, char_to_id, id_to_char = augment_with_pretrained(
        dico_chars_train.copy(),
        parameters['pre_char_emb'],
        list(itertools.chain.from_iterable(["".join([w[0] for w in s]) for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
    dico_chars_train = dico_chars

dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
# dico_pos, pos_to_id, id_to_pos = pos_mapping(train_sentences)

list_prefix = read_list(opts.dictionary)


# Index data
label_cnt = len(dico_tags)

train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, use_gaze, True, list_prefix= list_prefix, label_cnt= label_cnt, lower= lower, pos= pos
) # False:表示将data['gaze']one－hot向量全部置为0；True：不置0
#print "train_data[0]['gaze']:", train_data[0]['gaze']
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, use_gaze, True, list_prefix= list_prefix, label_cnt= label_cnt, lower= lower, pos= pos
)
# for data in dev_data:
#     data['pos_one_hot']
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, use_gaze, True, list_prefix= list_prefix, label_cnt= label_cnt, lower= lower, pos= pos
)

print "%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data))

# Save the mappings to disk
print 'Saving the mappings to disk...'
model.save_mappings(id_to_word, id_to_char, id_to_tag)#, id_to_pos)

# Build the model
f_train, f_eval, f_plot_cost = model.build(**parameters)
print 'yeh!'
# Reload previous model values
if opts.reload:
    print 'Reloading previous model...'
    model.reload()

#
# Train network
#
singletons = set([word_to_id[k] for k, v
                  in dico_words_train.items() if v == 1])
n_epochs = 150  # number of epochs over the training set
freq_eval = 1000  # evaluate on dev every freq_eval steps
best_dev = -np.inf
best_test = -np.inf
count = 0
# costfile = './cost_vec_' + str(parameters['word_dim']) + '_' + str(parameters['word_hidden_dim']) + str(parameters['L2_reg']) + '.txt'
# fw = codecs.open(costfile, 'w', 'utf-8')
# fw.write('epoch\t\ttrain_loss\t\tdev_los\t\ttest_loss\t\tdev_F1\t\ttest_F1\n')
F1_file = './vec_' + opts.tagger + "gaze_" + opts.dictionary + str(parameters['use_gaze']) + "char" + str(parameters['char_dim']) + "_" + str(parameters['char_hidden_dim']) \
          + "_word" + str(parameters['word_dim']) + "_" + str(parameters['word_hidden_dim']) + 'taggerhidden' + str(parameters['tagger_hidden_dim']) + '.txt'
fw = codecs.open(F1_file, 'w', 'utf-8')
fw.write("epoch\t\tdev_F1\t\ttest_F1\n")
for epoch in xrange(n_epochs):
    epoch_costs = []
    print "Starting epoch %i..." % epoch
        
    for i, index in enumerate(np.random.permutation(len(train_data))):
        count += 1
        input = create_input(train_data[index], parameters, True, use_gaze, pos, singletons)
        new_cost = f_train(*input)

        epoch_costs.append(new_cost)
        if i % 50 == 0 and i > 0 == 0:
            print "%i, cost average: %f" % (i, np.mean(epoch_costs[-50:]))
        if count % freq_eval == 0:
            #train_score = evaluate(parameters, f_eval, train_sentences,
            #                     train_data, id_to_tag, dico_tags)
            # train_cost = []
            # for i, data in enumerate(train_data):
            #     input = create_input(data, parameters, True, use_gaze, pos, singletons)
            #     train_cost.append(f_plot_cost(*input))
            # dev_cost = []
            # for i, data in enumerate(dev_data):
            #     input = create_input(data, parameters, True, use_gaze, pos, singletons)
            #     dev_cost.append(f_plot_cost(*input))
            # test_cost = []
            # for i, data in enumerate(test_data):
            #     input = create_input(data, parameters, True, use_gaze, pos, singletons)
            #     test_cost.append(f_plot_cost(*input))



	    print "dev:"
            dev_score = evaluate(parameters, f_eval, dev_sentences,
                                 dev_data, id_to_tag, tag_to_id, use_gaze, pos,  post_process, list_prefix, label_cnt)
	    print "test:"
            test_score = evaluate(parameters, f_eval, test_sentences,
                                  test_data, id_to_tag, tag_to_id, use_gaze, pos, post_process, list_prefix, label_cnt)

            fw.write('\t\t'.join([str(epoch), str(dev_score), str(test_score)]) + '\n')
            # cost_str = '\t\t'.join(
            #     [str(epoch), str(np.mean(train_cost)), str(np.mean(dev_cost)), str(np.mean(test_cost)), str(dev_score), str(test_score)])
            # fw.write(cost_str + '\n')

            #print "Score on train: %.5f" % train_score
            print "Score on dev: %.5f" % dev_score
            print "Score on test: %.5f" % test_score
            if dev_score > best_dev:
                best_dev = dev_score
                print "New best score on dev."
                print "Saving model to disk..."
                model.save()
            if test_score > best_test:
                best_test = test_score
                print "New best score on test."
    print "Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs))


print "cost time: %f" % (time.time() - start_time)
fw.close()
