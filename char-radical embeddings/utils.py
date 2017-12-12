#-*-coding:utf-8-*-
import os
import re
import codecs
import numpy as np
import theano
import theano.tensor as T
from time import strftime
from datetime import datetime


models_path = os.path.join(os.path.abspath('.'), "models")
eval_path = os.path.join(os.path.abspath('.'), "evaluation")
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")

def match_hash(key, buf, i, j, hash_words):
    for v in hash_words[key]:
        leng = len(v)
        rows, k = 0, 0

        while k < leng and j + rows < i:
            k += len(buf[j + rows])
            rows += 1
        if k == leng and j + rows <= i and "".join(buf[j:j + rows]) == v:
            return rows
    return 0

def read_list_data(str_chars, hash_words, label_cnt):
    gaze = [[0 for i in range(label_cnt)] for w in str_chars]
    i, j = len(str_chars), 0

    while j < i:
        rows = 0
        # if str_words[j] in hash_words:
        #     rows = match_hash(str_words[j], str_words, i, j, hash_words)
        # if rows == 0 and ' '.join(str_words[j]).split()[0] in hash_words:
        if ' '.join(str_chars[j]).split()[0] in hash_words:
            x = ' '.join(str_chars[j]).split()[0]
            rows = match_hash(x, str_chars, i, j, hash_words)
        if rows > 0:
            if rows == 1:
                gaze[j][0] = 1 #'S-ORG'
            else:
                gaze[j][1] = 1 #'B-ORG'
                for m in range(j + 1, j + rows - 1):
                    gaze[m][2] = 1 #'I-ORG'
                gaze[j + rows - 1][3] = 1 #'E-ORG'
            j += rows

        else:
            j += 1
    return gaze

def get_name(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in parameters.items():
        if type(v) is str and "/" in v:
            l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")


def set_values(name, param, pretrained):
    """
    Initialize a network parameter with pretrained values.
    We check that sizes are compatible.
    """
    param_value = param.get_value()
    if pretrained.size != param_value.size:
        raise Exception(
            "Size mismatch for parameter %s. Expected %i, found %i."
            % (name, param_value.size, pretrained.size)
        )
    param.set_value(np.reshape(
        pretrained, param_value.shape
    ).astype(np.float32))


def shared(shape, name):
    """
    Create a shared object of a numpy array.
    """
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return theano.shared(value=value.astype(theano.config.floatX), name=name)


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}

    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    s[0] = re.sub('\d', '0', s[0])
    return s



def iob2(tags):
    """
    Check that tags have a valid IOBES format.
    #Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
	split = tag.split('-')
	if len(split) != 2 or split[0] not in ['B', 'I', 'E', 'S']:
            return False
	
    return True


def ioesb1b2_iobes(tags):
    """
    IOESB1B2 -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B1':
            new_tags.append(tag.replace('B1-', 'B-'))
        elif tag.split('-')[0] == 'B2':
	    if i == 0:
		new_tags.append(tag.replace('B2-', 'B-'))
	    elif i > 0:
		if tags[i - 1].split('-')[0] != 'B1':
		    new_tags.append(tag.replace('B2-', 'B-'))
		elif tags[i - 1].split('-')[0] =='B1':
		    new_tags.append(tag.replace('B2-', 'I-'))
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag)
        else:
            raise Exception('Invalid IOESB1B2 format!')
    return new_tags


def ioesb1b2_iob(tags):
    """
    IOESB1B2 -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B1':
            new_tags.append(tag.replace('B1-', 'B-'))
	elif tag.split('-')[0] == 'B2':
	    if i == 0:
		new_tags.append(tag.replace('B2-', 'B-'))
	    elif i > 0:
		if (tags[i - 1].split('-')[0] != 'B1'):
		    new_tags.append(tag.replace('B2-', 'B-'))
	        elif tags[i - 1].split('-')[0] =='B1':
		    new_tags.append(tag.replace('B2-', 'I-'))
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags

def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_char_radicals(radicals):
    """
    Pad the radicals of the chars in a sentence.
    Input:
        - list of lists of ints (list of chars, a char being a list of radical indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where radicals are reversed)
        - list of ints corresponding to the index of the last radical of each character
    """
    max_length = max([len(rad) for rad in radicals])
    radical_for = []
    radical_rev = []
    radical_pos = []
    for rad in radicals:
        padding = [0] * (max_length - len(rad))
        radical_for.append(rad + padding)
        radical_rev.append(rad[::-1] + padding)
        radical_pos.append(len(rad) - 1)
    return radical_for, radical_rev, radical_pos


def create_input(data, parameters, add_label, use_gaze, singletons=None):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    chars = data['chars']
    radicals = data['radicals']
    if singletons is not None:
        chars = insert_singletons(chars, singletons)
    radical_for, radical_rev, radical_pos = pad_char_radicals(radicals)
    input = []
    if parameters['char_dim']:
        input.append(chars)
    if parameters['use_gaze']:
        gaze = data['gaze']
        input.append(gaze)
    if parameters['radical_dim']:
        input.append(radical_for)
        if parameters['radical_bidirect']:
            input.append(radical_rev)
        input.append(radical_pos)
    if add_label:
        input.append(data['tags'])
    return input

#evaluate(parameters, f_eval, dev_sentences,dev_data, id_to_tag, dico_tags)
def evaluate(parameters, f_eval, raw_sentences, parsed_sentences,
             id_to_tag, use_gaze, dictionary_tags):
    """
    Evaluate current model using CoNLL script.
    """
    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)
    
    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        input = create_input(data, parameters, False, use_gaze)

        if parameters['crf']:
            y_preds = np.array(f_eval(*input))[1:-1]
        else:
            y_preds = f_eval(*input).argmax(axis=1)  # f_eval(*input)的输出值是P矩阵，size为(n, n_tags)
        y_reals = np.array(data['tags']).astype(np.int32)
        assert len(y_preds) == len(y_reals)
        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
        r_tags = [id_to_tag[y_real] for y_real in y_reals]
        if parameters['tag_scheme'] == 'ioesb1b2':
            p_tags = ioesb1b2_iobes(p_tags)
            r_tags = ioesb1b2_iobes(r_tags)
	p_tags = iobes_iob(p_tags)
	r_tags = iobes_iob(r_tags)

	# print "len(p_tags): %d, len(r_tags): %d" % (len(p_tags), len(r_tags))
	# print "len(raw_sentence): %d" % (len(raw_sentence))
        for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
            new_line = " ".join(raw_sentence[i][:-1] + [r_tags[i], p_tags[i]])
            predictions.append(new_line)
            count[y_real, y_pred] += 1
        predictions.append("")

    # Write predictions to disk and run CoNLL script externally
    eval_id = np.random.randint(1000000, 2000000)
    output_path = os.path.join(eval_temp, "eval.%i.output" % eval_id)
    scores_path = os.path.join(eval_temp, "eval.%i.scores" % eval_id)
    with codecs.open(output_path, 'w', 'utf8') as f:
        f.write("\n".join(predictions))
    os.system("perl %s < %s > %s" % (eval_script, output_path, scores_path))
    print "current time is %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    #os.system("%s > %s" % (output_path, scores_path))

    # CoNLL evaluation results
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    for line in eval_lines:
        print line

    # Remove temp files
    #os.remove(output_path)
    #os.remove(scores_path)

    # Confusion matrix with accuracy for each tag
    print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in xrange(n_tags)] + ["Percent"])
    )
    for i in xrange(n_tags):
        print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
            str(i), id_to_tag[i], str(count[i].sum()),
            *([count[i][j] for j in xrange(n_tags)] +
              ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        )

    # Global accuracy
    print "%i/%i (%.5f%%)" % (
        count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
    )

    # F1 on all entities
    return float(eval_lines[1].strip().split()[-1])
    #numer = count.trace() - count[0][0]
    #demon = count[:, 1:].sum()

    #if demon != 0:
    #    print "precision:%i/%i (%.5f%%)" % (numer, demon, 100. * numer / demon )
    #    return 100. * numer / demon
    #else:
    #    print "precision:%i/%i (%.5f%%)" % (numer, demon, 0.)
    #    return 0.
        
def FOFE(char_weights, s_len, char_dim, context_size = 3, alpha = 0.5, direction = True):
    including = char_weights
    excluding = T.zeros_like(char_weights)
    
    if direction:
	for i in xrange(1, context_size):
	    including += np.power(alpha, i) * T.concatenate([np.zeros((i, char_dim)), char_weights[T.arange(s_len - i)]], axis = 0) 
	    excluding += np.power(alpha, i) * T.concatenate([np.zeros((i, char_dim)), char_weights[T.arange(s_len - i)]], axis = 0) 
    else:
	for i in xrange(1, context_size):
	    including += np.power(alpha, i) * T.concatenate([char_weights[T.arange(i, s_len)], np.zeros((i, char_dim))], axis = 0) 
	    excluding += np.power(alpha, i) * T.concatenate([char_weights[T.arange(i, s_len)], np.zeros((i, char_dim))], axis = 0) 

    #return theano.shared(value = including.astype(theano.config.floatX), name = str(direction) + '_' + 'including'), theano.shared(value = excluding.astype(theano.config.floatX), name = str(direction) + '_' + 'excluding') 
    return including, excluding		
