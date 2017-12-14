#-*-coding:utf-8-*-
import os
import re
import codecs
from collections import defaultdict
from utils import create_dico, create_mapping, zero_digits, read_list_data
from utils import iob, iob2, iob_iobes
import jieba

def read_list(dictionry_file):
    pattern = re.compile(r'^[A-Za-z]+$')

    hash_words = defaultdict(list)
    with codecs.open(dictionry_file, 'r', 'utf-8') as fr:
        for line in fr:
            line = line.strip()
            if len(line) == 0:
                continue
            elif len(pattern.findall(line)) > 0:
                # "英文公司名",for example
                if line.lower() not in hash_words[line.lower()]:
                    hash_words[line.lower()] += [line.lower()]
                    hash_words[line.lower()] += [line.lower() + unicode('公司', 'utf8')]
                    hash_words[line.lower()] += [line.lower() + unicode('集团', 'utf8')]
                    hash_words[line.lower()] += [line.lower() + unicode('资本', 'utf8')]
            elif ' ' in line:
                # "英文公司名"
                line = line.split()
                if ''.join(line).lower() not in hash_words[line[0].lower()]:
                    hash_words[line[0].lower()] += [''.join(line).lower()]
            else:
                # "将中文公司名全称作为索引"
                if line.lower() not in hash_words[line.lower()]:
                    hash_words[line.lower()] += [line.lower()]
                # "全模式切分"
                for x in jieba.cut(line, cut_all=True):
                    if line.lower() not in hash_words[x.lower()]:
                        hash_words[x.lower()] += [line.lower()]
                # "精确模式切分"
                for x in jieba.cut(line):
                    if line.lower() not in hash_words[x.lower()]:
                        hash_words[x.lower()] += [line.lower()]
                # 将中文公司名中的字作为索引
                for x in ' '.join(line).split():
                    if line.lower() not in hash_words[x.lower()]:
                        hash_words[x.lower()] += [line.lower()]

    for k in hash_words.keys():
        hash_words[k].sort(key=lambda x: len(x), reverse=True)

    return hash_words


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        if not line.rstrip():
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = zero_digits(line.rstrip().split()) if zeros else line.rstrip().split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB,IOBES and IOESB1B2 schemes are accepted.
    """

    for i, s in enumerate(sentences):
            tags = [w[-1] for w in s]
            # Check that tags are given in the IOB format
            if not iob(tags):
		s_str = '\n'.join(' '.join(w) for w in s)
		print s_str
                #raise Exception('Sentences should be given in IOESB1B2 format! ' +
                #                'Please check sentence %i:\n%s' % (i, s_str))
                raise Exception('Sentences should be given in IOBES format! ' +
                                'Please check sentence %i:\n%s' % (i, s_str))
            if tag_scheme == 'ioesb1b2':
                for word, new_tag in zip(s, tags):
                    word[-1] = new_tag
            elif tag_scheme == 'iob':
                new_tags = tags_to_iob(tags)
                for word, new_tag in zip(s, new_tags):
                    word[-1] = new_tag
            elif tag_scheme == 'iobes':
                #new_tags = tags_to_iobes(tags)
                for word, new_tag in zip(s, tags):
                    word[-1] = new_tag
            else:
                raise Exception('Unknown tagging scheme!')

def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print "Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    )
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print "Found %i unique characters" % len(dico)
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print "Found %i unique named entity tags" % len(dico)
    return dico, tag_to_id, id_to_tag

def pos_mapping(sentences):
    """
    Create a dictionary and a mapping of pos, sorted by frequency.
    """
    pos = [[word[1] for word in s] for s in sentences]
    dico = create_dico(pos)
    pos_to_id, id_to_pos = create_mapping(dico)
    print "Found %i unique pos" % len(dico)
    return dico, pos_to_id, id_to_pos


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    #caps = [cap_feature(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars
    #    'caps': caps
    }


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, use_gaze, trainset, list_prefix= None, label_cnt= None, lower=False, pos= False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - pos indexes
        - gaze indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]

        if pos:
            poses = [w[1] for w in s]
            # for w in s:
            #     if len(w) > 1:
            #         print w[1]
                    
            pos_one_hot = [[0 for i in range(6)] for w in s]
            for i, p in enumerate(poses):
                if p == 'n' or p == 'ns' or p == 'nt':
                    pos_one_hot[i][0] = 1
                elif p == 'nx':
                    pos_one_hot[i][1] = 1
                elif p == 'nz':
                    pos_one_hot[i][2] = 1
                elif p == 'w':
                    pos_one_hot[i][3] = 1
                elif p == 'v' or p == 'vi' or p == 'vn':
                    pos_one_hot[i][4] = 1
                else:
                    pos_one_hot[i][5] = 1
        if use_gaze:
            gaze = read_list_data(str_words, trainset, list_prefix, label_cnt)
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]


        tags = [tag_to_id[w[-1]] for w in s]

        if use_gaze and pos:
            data.append({
                'str_words': str_words,
                'words': words,
                'chars': chars,
                'gaze': gaze,
                'pos_one_hot': pos_one_hot,
                'tags': tags,
            })
        elif use_gaze:
            data.append({
                'str_words': str_words,
                'words': words,
                'chars': chars,
                'gaze': gaze,
                'tags': tags,
            })
        elif pos:
            data.append({
                'str_words': str_words,
                'words': words,
                'chars': chars,
                'pos_one_hot': pos_one_hot,
                'tags': tags,
            })
        else:
            data.append({
                'str_words': str_words,
                'words': words,
                'chars': chars,
                'tags': tags,
            })


    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print 'Loading pretrained embeddings from %s...' % ext_emb_path
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8', 'ignore')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word
