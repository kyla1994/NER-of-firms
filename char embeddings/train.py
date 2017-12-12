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

from utils import models_path, evaluate, eval_temp, eval_script
from loader import char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset, read_list
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
    "-c", "--char_dim", default="300",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_hidden_dim", default="300",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-R", "--layer2_hidden_dim", default="50",
    type='int', help="The hidden layer size of the second LSTM/GRU layer"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM/GRU for chars"
)
optparser.add_option(
    "-p", "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="0",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-g", "--dictionary", default="",
    help="Gaze dictionary"
)
optparser.add_option(
    "-U", "--use_gaze", default="0",
    type='int', help="Use gaze"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-E", "--layer2", default="1",
    type='int', help="The number of layers (1 represents 2 layers)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-B", "--batch_size", default="1",
    type='int', help="The batch size"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.005",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-e", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
opts = optparser.parse_args()[0]

# Parse parameters
parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_hidden_dim'] = opts.char_hidden_dim
parameters['layer2'] = opts.layer2
parameters['layer2_hidden_dim'] = opts.layer2_hidden_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['use_gaze'] = opts.use_gaze == 1
parameters['batch_size'] = opts.batch_size
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
#parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method

# Check parameters validity
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert parameters['char_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes', 'ioesb1b2']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['char_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Initialize model
start_time = time.time() 
model = Model(parameters=parameters, models_path=models_path)
print "Model location: %s" % model.model_path

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']
use_gaze = parameters['use_gaze']
batch_size = opts.batch_size


# Load sentences
train_sentences = loader.load_sentences(opts.train, zeros, lower)
dev_sentences = loader.load_sentences(opts.dev, zeros, lower)
test_sentences = loader.load_sentences(opts.test,zeros, lower)

# Use selected tagging scheme (IOB / IOESB / IOESB1B2)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

# Create a dictionary / mapping of chars
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_emb']:
    dico_chars_train = char_mapping(train_sentences)[0]
    dico_chars, char_to_id, id_to_char = augment_with_pretrained(
        dico_chars_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable([[w[0] for w in s] for s in test_sentences + dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
    dico_chars_train = dico_chars

dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
dico_tags_train = dico_tags


list_prefix = read_list(opts.dictionary)
# Index data
label_cnt = len(dico_tags)
train_data = prepare_dataset(
    train_sentences, char_to_id, tag_to_id, lower, list_prefix= list_prefix, label_cnt= label_cnt
)
dev_data = prepare_dataset(
    dev_sentences, char_to_id, tag_to_id, lower, list_prefix= list_prefix, label_cnt= label_cnt
)
test_data = prepare_dataset(
    test_sentences, char_to_id, tag_to_id, lower, list_prefix= list_prefix, label_cnt= label_cnt
)

print "%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data))

# Save the mappings to disk
print 'Saving the mappings to disk...'
model.save_mappings(id_to_char, id_to_tag)

# Build the model
f_train, f_eval = model.build(**parameters)

# Reload previous model values
if opts.reload:
    print 'Reloading previous model...'
    model.reload()

#
# Train network
#
singletons = set([char_to_id[k] for k, v
                  in dico_chars_train.items() if v == 1])
n_epochs = 150  # number of epochs over the training set
freq_eval = 1000  # evaluate on dev every freq_eval steps
best_dev = -np.inf
best_test = -np.inf
count = 0

F1_file = "layer2_" + str(opts.layer2) + "gaze_" + str(opts.use_gaze) + '_' + str(opts.char_dim) + "_" + str(opts.char_hidden_dim) + "_" + str(opts.layer2_hidden_dim) + ".txt"
fw = codecs.open(F1_file, 'w', 'utf-8')
fw.write("epoch\t\t\tdev_F1\t\ttest_F1\n")
for epoch in xrange(n_epochs):
    epoch_costs = []
    print "Starting epoch %i..." % epoch

    for i, index in enumerate(np.random.permutation(len(train_data))):
        count += 1
        input = create_input(train_data[index], parameters, True, singletons)
        new_cost = f_train(*input)

        epoch_costs.append(new_cost)
        if i % 50 == 0 and i > 0 == 0:
            print "%i, cost average: %f" % (i, np.mean(epoch_costs[-50:]))
        if count % freq_eval == 0:
#            train_score = evaluate(parameters, f_eval, train_sentences,
#                                 train_data, id_to_tag, dico_tags)
	    
            dev_score = evaluate(parameters, f_eval, dev_sentences,
                                 dev_data, id_to_tag, use_gaze, dico_tags)
            test_score = evaluate(parameters, f_eval, test_sentences,
                                  test_data, id_to_tag, use_gaze, dico_tags)

            fw.write("\t\t".join([str(epoch), str(dev_score), str(test_score)]) + '\n')
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
