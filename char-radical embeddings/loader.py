#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
from collections import defaultdict
import re
import jieba
import codecs
from utils import create_dico, create_mapping, zero_digits, read_list_data
from utils import iob2, iob_iobes

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

def load_sentences(path, zeros, lower):
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
	    #word = ['!' if w in ('-',',','，','。','.','>','?', ':', '：') else w for w in word]
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
            if not iob2(tags):
		s_str = '\n'.join(' '.join(w) for w in s)
		print s_str
                raise Exception('Sentences should be given in IOBES format! ' +
                                'Please check sentence %i:\n%s' % (i, s_str))
            if tag_scheme == 'iobes':
                for word, new_tag in zip(s, tags):
                    word[-1] = new_tag
            elif tag_scheme == 'iob':
                new_tags = tags_to_iob(tags)
                for word, new_tag in zip(s, new_tags):
                    word[-1] = new_tag
            else:
                raise Exception('Unknown tagging scheme!')

def char_mapping(sentences):
    """
    Create a dictionary and a mapping of chars, sorted by frequency.
    """
    chars = [[x[0] for x in s] for s in sentences]
    dico = create_dico(chars)
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print "Found %i unique chars (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    )
    return dico, char_to_id, id_to_char

def radical_mapping(sentences, radicals):
    """
    Create a dictionary and a mapping of radicals, sorted by frequency.
    """
    tmp = [[radicals[x[0]] if x[0] in radicals else ['<UNK>'] for x in s] for s in sentences]
    rads = [["".join(xx) for xx in xs] for xs in tmp]
    dico = create_dico(rads)
    # dico['<UNK>'] = 10000000
    radical_to_id, id_to_radical = create_mapping(dico)
    print "Found %i unique radicals (%i in total)" % (
        len(dico), sum(len(x) for x in rads))
    return dico, radical_to_id, id_to_radical

def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print "Found %i unique named entity tags" % len(dico)
    return dico, tag_to_id, id_to_tag


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


def prepare_sentence(str_chars, char_to_id, radical_to_id, radicals_dict, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
             for w in str_chars]
    #radicals = [[radical_to_id[r] if r in radical_to_id else [] for r in radicals_dict[c] ]
    #         for c in str_chars if c in radicals_dict]
    radicals = []
    for c in str_chars:
        tmp = []
        if c in radicals_dict:
            tmp.append([radical_to_id[r] if r in radical_to_id else radical_to_id['<UNK>']] for r in radicals_dict[c])
        tmp.append([-1])
        radicals.append(tmp)
    return {
        'str_chars': str_chars,
        'chars': chars,
        'radicals': radicals
    }


def prepare_dataset(sentences, char_to_id, radical_to_id, tag_to_id, use_gaze,radicals_dict, lower=False, list_prefix= None, label_cnt= None):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - char indexes
        - radical indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for i, s in enumerate(sentences):
        str_chars = [w[0] for w in s]
        chars = [char_to_id[w if w in char_to_id else '<UNK>']
                 for w in str_chars]
        #radicals = [[radical_to_id[r] for r in radicals_dict[c]]
        #         for c in str_chars if c in radicals_dict]
        radicals = []
        for c in str_chars:
            if c in radicals_dict:
                radicals.append([radical_to_id[r] if r in radical_to_id else radical_to_id['<UNK>'] for r in radicals_dict[c]])
            else:
                radicals.append([-1])
        if use_gaze:
            gaze = read_list_data(str_chars, list_prefix, label_cnt)
        tags = [tag_to_id[w[-1]] for w in s]
        if use_gaze:
            data.append({
                'str_chars': str_chars,
                'chars': chars,
                'radicals': radicals,
                'gaze': gaze,
                'tags': tags,
            })
        else:
            data.append({
                'str_chars': str_chars,
                'chars': chars,
        	'radicals': radicals,
                'tags': tags,
            })
    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with chars that have a pretrained embedding.
    If `chars` is None, we add every char that has a pretrained embedding
    to the dictionary, otherwise, we only add the chars that are given by
    `chars` (typically the chars in the development and test sets.)
    """
    print 'Loading pretrained embeddings from %s...' % ext_emb_path
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8', 'ignore')
        if len(ext_emb_path) > 0
    ])

    # We either add every char in the pretrained file,
    # or only chars given in the `chars` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    char_to_id, id_to_char = create_mapping(dictionary)
    return dictionary, char_to_id, id_to_char
