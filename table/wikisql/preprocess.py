# -*- coding: utf-8 -*-
import json
import babel
from babel import numbers
from babel.numbers import parse_decimal, NumberFormatError

import nltk
import os
import unicodedata
import re
import time

import nsm
from nsm import word_embeddings
from nsm import data_utils

import tensorflow as tf

# FLAGS
FLAGS = tf.app.flags.FLAGS  
tf.flags.DEFINE_string('raw_input_dir', '',
                       'path.')
tf.flags.DEFINE_string('processed_input_dir', '',
                       'path to the folder to save all the processed data.')
tf.flags.DEFINE_integer('n_train_shard', 30, '.')


# Copied from utils to avoid relative import.
# [TODO] Use a cleaner solution.
def find_tk_in_model(tk, model):
    special_tk_dict = {'-lrb-': '(', '-rrb-': ')'}
    if tk in model:
        return [tk]
    elif tk in special_tk_dict:
        return [special_tk_dict[tk]]
    elif tk.upper() in model:
        return [tk.upper()]
    elif tk[:1].upper() + tk[1:] in model:
        return [tk[:1].upper() + tk[1:]]
    elif re.search('\\/', tk):
        tks = tk.split('\\\\/')
        if len(tks) == 1:
          return []
        valid_tks = []
        for tk in tks:
            valid_tk = find_tk_in_model(tk, model)
            if valid_tk:
                valid_tks += valid_tk
        return valid_tks
    else:
        return []


def tokens_contain(string_1, string_2):
    tks_1 = nltk.tokenize.word_tokenize(string_1)
    tks_2 = nltk.tokenize.word_tokenize(string_2)
    return set(tks_2).issubset(set(tks_1))


# preprocess the questions.
def string_in_table(string, kg):
    # print string
    for k, node in kg['kg'].iteritems():
        for prop, val in node.iteritems():
            if (isinstance(val[0], str) or isinstance(val[0], unicode)) and string in val[0]:
                return True
    else:
        return False


def num_in_table(ent, kg):
    props_set = set(kg['num_props'])
    for k, node in kg['kg'].iteritems():
        for prop, val in node.iteritems():
            if prop in props_set and ent == val:
                return True
    else:
        return False        


def prop_in_question_score(prop, example, stop_words, binary=True):
    question = example['question']
    prop = prop[2:]
    prop = u'-'.join(prop.split(u'-')[:-1])
    prop_tks = prop.split(u'_')
    n_in_question = 0
    for tk in prop_tks:
        # tk = tk.decode('utf-8')
        if tk not in stop_words and tk in question:
            n_in_question += 1
    if binary:
        n_in_question = min(n_in_question, 1)
    return n_in_question


def expand_entities(e, table_dict):
    ents = [ent for ent in e['entities']
            if ent['type'] == 'string_list' and ent['value'][0]]
    other_ents = [ent for ent in e['entities'] if ent['type'] != 'string_list']
    kg = table_dict[e['context']]
    l = len(ents)
    new_ents = []
    i = 0
    tokens = e['tokens']
    for ent in ents:
        # relies on order. 
        if new_ents and ent['token_end'] <= new_ents[-1]['token_end']:
            continue
        else:
            ent['value'][0] = tokens[ent['token_start']]
            new_ents.append(ent)
            while True and ent['token_end'] < len(tokens):
                new_str_list = (
                  [s.join([to_unicode(ent['value'][0]),
                           to_unicode(tokens[ent['token_end']])]) 
                   for s in [u' ', u'-', u'']] + 
                  [s.join([to_unicode(ent['value'][0]),
                           to_unicode(normalize(tokens[ent['token_end']]))])
                   for s in [u' ', u'-', u'']] + 
                  [s.join([to_unicode(normalize(ent['value'][0])),
                           to_unicode(tokens[ent['token_end']])]) 
                   for s in [u' ', u'-', u'']] + 
                  [s.join([to_unicode(normalize(ent['value'][0])),
                           to_unicode(normalize(tokens[ent['token_end']]))])
                   for s in [u' ', u'-', u'']])
                for new_str in new_str_list:
                  if string_in_table(new_str, kg):
                    ent['token_end'] += 1
                    ent['value'] = [new_str]
                    break
                else:
                  break
            ent['value'] = [normalize(ent['value'][0])]
    e['entities'] = new_ents + other_ents        


def to_unicode(tk, encoding='utf-8'):
  if isinstance(tk, str):
    return tk.decode(encoding)
  else:
    return tk


def normalize(x):
    if not isinstance(x, unicode):
        x = x.decode('utf8', errors='ignore')
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(ur"[‘’´`]", "'", x)
    x = re.sub(ur"[“”]", "\"", x)
    x = re.sub(ur"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(ur"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(ur"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(ur'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(ur'\s+', ' ', x, flags=re.U).lower().strip()
    return x
        

def process_prop(prop):
    return prop.replace(' ', '_').lower()


def table2kg(table):
    tkg = {}
    tkg[u'datetime_props'] = []
    num_props = []
    props = []
    for p, tp in zip(table['header'], table['types']):
        prop = process_prop(p)
        if tp == 'real':
            prop = u'r.{}-number'.format(prop)
            num_props.append(prop)
        else:
            prop = u'r.{}-string'.format(prop)
        props.append(prop)
    tkg[u'props'] = props
    tkg[u'num_props'] = num_props
    tkg[u'name'] = table[u'id']
    rows = table[u'rows']
    tkg[u'row_ents'] = [u'row_{}'.format(i) for i in xrange(len(rows))]
    kg = {}
    for i, row in enumerate(rows):
        processed_vals = []
        for j, val in enumerate(row):
            if table['types'][j] == 'real':
                if isinstance(val, int) or isinstance(val, float):
                    processed_vals.append([val])
                else:
                    processed_vals.append([float(babel.numbers.parse_decimal(val))])
            else:
                processed_vals.append([val.lower()])
        kg[u'row_{}'.format(i)] = dict(zip(props, processed_vals))
    tkg[u'kg'] = kg
    return tkg


def annotate_question(q, id, kg_dict, stop_words):
    tokens = nltk.tokenize.word_tokenize(q['question'])
    tokens = [tk.lower() for tk in tokens]
    e = {}
    e['tokens'] = tokens
    e['question'] = q['question']
    e['context'] = q['table_id']
    e['sql'] = q['sql']
    e['answer'] = q['answer']
    e['id'] = id
    e['entities'] = []
    e['in_table'] = [0] * len(tokens)
    # entities are normalized tokens
    e['processed_tokens'] = tokens[:]
    kg = kg_dict[q['table_id']]
    for i, tk in enumerate(tokens):
        if tk not in stop_words:
            if string_in_table(normalize(tk), kg):
                e['entities'].append(
                    dict(value=[normalize(tk)], token_start=i, token_end=i+1,
                         type='string_list'))
                e['in_table'][i] = 1
        try:
            val = float(babel.numbers.parse_decimal(tk))
            if val is not None:
                e['entities'].append(
                    dict(value=[val], token_start=i, token_end=i+1,
                         type='num_list'))
                if num_in_table(val, kg):
                    e['in_table'][i] = 1
                e['processed_tokens'][i] = '<{}>'.format('NUMBER')
        except NumberFormatError:
            pass
    e['features'] = [[it] for it in e['in_table']]
    e['prop_features'] = dict(
        [(prop, [prop_in_question_score(
            prop, e, stop_words, 
            binary=False)])
         for prop in kg['props']])
    return e


def create_vocab(examples, embedding_model, min_count):
    token_count = {}
    for e in examples:
      for tk in e['tokens']:
        # Token must be in glove and also appears more than min_count.
        if find_tk_in_model(tk, embedding_model):
          try:
            token_count[tk] += 1
          except KeyError:
            token_count[tk] = 1
    en_vocab = data_utils.generate_vocab_from_token_count(
      token_count, min_count=min_count)
    return en_vocab


def dump_examples(examples, fn):
    t1 = time.time()
    with open(fn, 'w') as f:
        for i, e in enumerate(examples):
            f.write(json.dumps(e))
            f.write('\n')
    t2 = time.time()
    print '{} sec used dumping {} examples.'.format(t2 - t1, len(examples))
 

def main(unused_argv):
    assert tf.gfile.Exists(FLAGS.raw_input_dir)
    if not tf.gfile.Exists(FLAGS.processed_input_dir):
        tf.gfile.MkDir(FLAGS.processed_input_dir)

    table_file = os.path.join(FLAGS.processed_input_dir, 'tables.jsonl')
    stop_words_file = os.path.join(FLAGS.raw_input_dir, 'stop_words.json')

    with open(stop_words_file, 'r') as f:
        stop_words = json.load(f)

    # Load datasets. 
    train_set = []
    with open(os.path.join(FLAGS.raw_input_dir, 'train.jsonl'), 'r') as f:
        for line in f:
            train_set.append(json.loads(line))

    dev_set = []
    with open(os.path.join(FLAGS.raw_input_dir, 'dev.jsonl'), 'r') as f:
        for line in f:
            dev_set.append(json.loads(line))

    test_set = []
    with open(os.path.join(FLAGS.raw_input_dir, 'test.jsonl'), 'r') as f:
        for line in f:
            test_set.append(json.loads(line))

    # Load tables.
    train_table_dict = {}
    with open(os.path.join(FLAGS.raw_input_dir, 'train.tables.jsonl'), 'r') as f:
        for line in f:
            _table = json.loads(line)
            train_table_dict[_table['id']] = _table

    dev_table_dict = {}
    with open(os.path.join(FLAGS.raw_input_dir, 'dev.tables.jsonl'), 'r') as f:
        for line in f:
            _table = json.loads(line)
            dev_table_dict[_table['id']] = _table

    test_table_dict = {}
    with open(os.path.join(FLAGS.raw_input_dir, 'test.tables.jsonl'), 'r') as f:
        for line in f:
            _table = json.loads(line)
            test_table_dict[_table['id']] = _table

    # Collect all the tables.
    print 'Start collecting all the tables.'
    kg_dict = {}
    for tb_dict in [dev_table_dict, train_table_dict, test_table_dict]:
        for i, (k, v) in enumerate(tb_dict.iteritems()):
            if i % 1000 == 0:
                print i
            kg_dict[k] = table2kg(v)

    # Check if the string or number value has the correct type. 
    for kg in kg_dict.values():
        for _, v in kg['kg'].iteritems():
            for prop, val in v.iteritems():
                if prop[-7:] == '-number':
                    for num in val:
                        if not (isinstance(num, int) or isinstance(num, float)):
                            print kg
                            raise ValueError
                if prop[-7:] == '-string':
                    for num in val:
                        if not isinstance(num, unicode):
                            print kg
                            raise ValueError

    # Save the tables. 
    with open(table_file, 'w') as f:
        for _, v in kg_dict.iteritems():
            f.write(json.dumps(v) + '\n')

    # Load the gold answers.
    with open(os.path.join(FLAGS.raw_input_dir, 'dev_gold.json'), 'r') as f:
        dev_answers = json.load(f)

    for q, ans in zip(dev_set, dev_answers):
        q['answer'] = ans

    with open(os.path.join(FLAGS.raw_input_dir, 'train_gold.json'), 'r') as f:
        train_answers = json.load(f)

    for q, ans in zip(train_set, train_answers):
        q['answer'] = ans

    with open(os.path.join(FLAGS.raw_input_dir, 'test_gold.json'), 'r') as f:
        test_answers = json.load(f)

    for q, ans in zip(test_set, test_answers):
        q['answer'] = ans

    # Annotate the examples and dump to files.
    train_split_jsonl = os.path.join(
        FLAGS.processed_input_dir, 'train_split.jsonl')
    dev_split_jsonl = os.path.join(
        FLAGS.processed_input_dir, 'dev_split.jsonl')
    test_split_jsonl = os.path.join(
        FLAGS.processed_input_dir, 'test_split.jsonl')

    t1 = time.time()
    dev_examples = []
    print 'start annotating dev examples.'
    for i, q in enumerate(dev_set):
        if i % 500 == 0:
            print i
        e = annotate_question(q, 'dev-{}'.format(i), kg_dict, stop_words)
        expand_entities(e, kg_dict)
        dev_examples.append(e)
    t2 = time.time()
    print '{} sec used annotating dev examples.'.format(t2 - t1)
    dump_examples(dev_examples, dev_split_jsonl)

    t1 = time.time()
    train_examples = []
    print 'start annotating train examples.'
    for i, q in enumerate(train_set):
        if i % 500 == 0:
            print i
        e = annotate_question(q, 'train-{}'.format(i), kg_dict, stop_words)
        expand_entities(e, kg_dict)
        train_examples.append(e)
    t2 = time.time()
    print '{} sec used annotating train examples.'.format(t2 - t1)
    dump_examples(train_examples, train_split_jsonl)


    t1 = time.time()
    test_examples = []
    print 'start annotating test examples.'
    for i, q in enumerate(test_set):
        if i % 500 == 0:
            print i
        e = annotate_question(q, 'test-{}'.format(i), kg_dict, stop_words)
        expand_entities(e, kg_dict)
        test_examples.append(e)
    t2 = time.time()
    print '{} sec used annotating test examples.'.format(t2 - t1)
    dump_examples(test_examples, test_split_jsonl)

    train_shards = []
    for i in range(FLAGS.n_train_shard):
        train_shards.append([])
    for i, e in enumerate(train_examples):
        train_shards[i % FLAGS.n_train_shard].append(e)

    for i, sh in enumerate(train_shards):
        train_shard_jsonl = os.path.join(
            FLAGS.processed_input_dir, 'train_split_shard_{}-{}.jsonl'.format(
                FLAGS.n_train_shard, i))
        dump_examples(sh, train_shard_jsonl)

    # Load pretrained embeddings.
    vocab_file = os.path.join(
        FLAGS.raw_input_dir, "wikisql_glove_vocab.json")
    embedding_file = os.path.join(
        FLAGS.raw_input_dir, "wikisql_glove_embedding_mat.npy")
    embedding_model = word_embeddings.EmbeddingModel(vocab_file, embedding_file)

    for i in xrange(1, 11):
        en_vocab = create_vocab(
          train_examples + dev_examples, embedding_model, i)
        vocab_file = os.path.join(
          FLAGS.processed_input_dir, "en_vocab_min_count_{}.json".format(i))
        with open(vocab_file, 'w') as f:
          json.dump(en_vocab.vocab, f, sort_keys=True, indent=2)
        print 'min_tk_count: {}, vocab size: {}'.format(i, len(en_vocab.vocab))    


if __name__ == '__main__':  
    tf.app.run()
