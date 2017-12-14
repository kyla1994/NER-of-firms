#!/usr/bin/env python

import os
import time
import codecs
import optparse
import numpy as np
from loader import prepare_sentence, prepare_dataset, read_list
from utils import create_input, iobes_iob, zero_digits, post_processes, read_list_data
from model import Model

optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--model", default="",
    help="Model location"
)
optparser.add_option(
    "-S", "--post_process", default="0",
    type='int', help="Use post-process"
)
optparser.add_option(
    "-i", "--input", default="",
    help="Input file location"
)
optparser.add_option(
    "-o", "--output", default="",
    help="Output file location"
)
optparser.add_option(
    "-d", "--delimiter", default=" ",
    help="Delimiter to separate words from their tags"
)
opts = optparser.parse_args()[0]

# Check parameters validity
assert opts.delimiter
assert os.path.isdir(opts.model)
assert os.path.isfile(opts.input)

# Load existing model
print "Loading model..."
model = Model(model_path=opts.model)
parameters = model.parameters

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]
id_to_word, id_to_char, id_to_tag = [
    {k: v for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]
# Load the model


_, f_eval, _ = model.build(training=False, **parameters)
model.reload()


list_prefix = read_list(parameters['dictionary'])
label_cnt = len(tag_to_id)

f_output = codecs.open(opts.output, 'w', 'utf-8')
start = time.time()

print 'Tagging...'
with codecs.open(opts.input, 'r', 'utf-8') as f_input:
    count = 0
    for line in f_input:
        str_words = line.rstrip().split()
        if line:
            # Lowercase sentence
            if parameters['lower']:
                line = line.lower()
            # Replace all digits with zeros
            if parameters['zeros']:
                line = zero_digits(line)
            # Prepare input
            def f(x): return x.lower() if parameters['lower'] else x
            words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                     for w in str_words]
            chars = [[char_to_id[c] for c in w if c in char_to_id]
                     for w in str_words]
            data = {
                'str_words': str_words,
                'words': words,
                'chars': chars
            }

            if parameters['use_gaze']:
                gaze = read_list_data(str_words, True, list_prefix, label_cnt)
                data['gaze'] = gaze

            input = create_input(data, parameters, False, parameters['use_gaze'], parameters['POS'])
            # Decoding
            if parameters['tagger'] == 'crf':
                y_preds = np.array(f_eval(*input))[1:-1]
            else:
                y_preds = f_eval(*input).argmax(axis=1)

            if opts.post_process:
                y_preds = post_processes(input, y_preds, tag_to_id, id_to_tag)

            y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]
            # Output tags in the IOB2 format
            if parameters['tag_scheme'] == 'iobes':
                y_preds = iobes_iob(y_preds)
            # Write tags
            assert len(y_preds) == len(words)
            f_output.write('%s\n' % '\n'.join('%s%s%s' % (w, opts.delimiter, y)
                                             for w, y in zip(str_words, y_preds)))
            f_output.write('\n')
        else:
            f_output.write('\n')
        count += 1
        if count % 100 == 0:
            print count

print '---- %i lines tagged in %.4fs ----' % (count, time.time() - start)
f_output.close()
