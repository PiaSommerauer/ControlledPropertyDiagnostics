import numpy as np
import inflect
from sklearn.metrics import pairwise_distances
import csv
import os
import json


def load_prop_set(prop):
    path = f'../data/aggregated/{prop}.json'
    with open(path) as infile:
        prop_dict = json.load(infile)
    return prop_dict


# def get_props():
#     path_dir = '../data/aggregated_vocab/'
#     files = os.listdir(path_dir)
#     props = [p.split('.')[0] for p in files]
#     return props


def load_seed(prop, model_dist, split_name, n):
    path = f'../data/train_test_splits/random-words-seed-0/giga-google-wiki/{prop}/seed.text'
    with open(path) as infile:
        text = infile.read().strip()
    seed = text.split(': ')[-1]
    return seed


def load_set(prop, model_dist_name, split_name, data_split, syn_control=False, pl_extension=False):
    path_dir = f'../data/train_test_splits/{split_name}/{model_dist_name}/{prop}/'
    path_f = f'{path_dir}{data_split}-syn_control_{syn_control}-pl_extension_{pl_extension}.csv'
    with open(path_f) as infile:
        data = list(csv.DictReader(infile))
    return data


def seed_chain_to_file(seed_chain, model_dist_name, split_name, n, prop):
    path = f'../data/train_test_splits/random-words-seed-{n}/giga-google-wiki/{prop}/seed_chain.csv'
    with open(path, 'w') as outfile:
        outfile.write('word1,word2,cos\n')
        for (w1, w2), cos in seed_chain.items():
            outfile.write(f'{w1},{w2},{cos}\n')


def get_matrix(concepts, model):
    engine = inflect.engine()
    matrix = []
    oov = []
    words = []
    for w in concepts:
        if w in model.vocab:
            vec = model[w]
            matrix.append(vec)
            words.append(w)
        else:
            oov.append(w)
            # pluralize:
            # plural = engine.plural(w)
            # if plural in model.vocab:
            #  vec = model[plural]
            # matrix.append(vec)
            # words.append(plural)
            # else:

    matrix = np.array(matrix)
    return matrix, words, oov


def get_closest_word_cos(target_word, words, word_index_dict, matrix):
    i_target = word_index_dict[target_word]
    vec_target = matrix[i_target]
    dists = pairwise_distances(matrix, [vec_target], metric='cosine')
    indices_sorted = np.argsort(dists.flatten())

    return indices_sorted, dists


def get_seed_chain(model, concepts, seed, n_pos):
    matrix, words, oov = get_matrix(concepts, model)
    word_index_dict = dict()
    for n, w in enumerate(words):
        word_index_dict[w] = n

    # pick seed that is in model vocab

    words_selected = set()
    words_selected.add(seed)
    word_pair_cosines = dict()
    target_word = seed

    search_words = True
    while search_words:
        indices_sorted, dists = get_closest_word_cos(target_word, words, word_index_dict, matrix)
        for i in indices_sorted:
            w = words[i]
            cos = dists[i][0]
            if w not in words_selected:
                words_selected.add(w)
                word_pair_cosines[(target_word, w)] = cos
                target_word = w
                break
            else:
                continue
        words_available = [w for w in words if w not in words_selected]
        if len(words_selected) == n_pos:
            search_words = False
            print('found enough')
        elif len(words_available) == 0:
            search_words = False
            print('no more candidates')
    return word_pair_cosines


def get_label_distribution_set(prop, data_split, model_dist):
    if data_split in ['random-words-seed', 'random-words-no-dist']:
        data_splits = [f'{data_split}-{n}' for n in range(10)]
    else:
        data_splits = [data_split]

    props_same = []

    for data_split in data_splits:
        path = f'../data/train_test_splits/{data_split}/{model_dist}/{prop}/'
        test = 'test-syn_control_False-pl_extension_False.csv'
        train = 'train-syn_control_False-pl_extension_False.csv'

        data = []
        for s in [train, test]:
            full_path = f'{path}{s}'
            with open(full_path) as infile:
                data.extend(list(csv.DictReader(infile)))

        # get words pos and neg:
        words_pos = [d['word'] for d in data if d['label'] == 'pos']
        words_neg = [d['word'] for d in data if d['label'] == 'neg']

        original_labels = dict()
        path_original = f'../data/aggregated/{prop}.json'
        # print(path_original)
        with open(path_original) as infile:
            prop_d = json.load(infile)
        for c, d in prop_d.items():
            label = d['ml_label']
            if label in ['all', 'all-some', 'some', 'few-some']:
                original_labels[c] = 'pos'
            elif label in ['few']:
                original_labels[c] = 'neg'
            else:
                original_labels[c] = None

        original_labels_pos = [original_labels[w] for w in words_pos if w in original_labels]
        original_labels_neg = [original_labels[w] for w in words_neg if w in original_labels]

        prop_pos = original_labels_pos.count('pos') / len(original_labels_pos)
        prop_neg = original_labels_pos.count('neg') / len(original_labels_pos)
        prop_bigger = max([prop_pos, prop_neg])
        props_same.append(prop_bigger)
    return sum(props_same) / len(props_same)


def get_label_dist(set_type, model_dist):
    properties = get_props(set_type=set_type)

    data_splits = ['random-words-seed', 'random-words-no-dist']  # 'random-words', 'standard-cosine']

    label_data = []
    for prop in properties:
        d = dict()
        d['property'] = prop
        for data_split in data_splits:
            prop_mean = get_label_distribution_set(prop, data_split, model_dist)
            d[data_split] = prop_mean
        label_data.append(d)
    # get means

    mean_d = dict()
    mean_d['property'] = 'mean'
    for datasplit in data_splits:
        values = [d[datasplit] for d in label_data]
        mean = sum(values) / len(values)
        mean_d[datasplit] = mean
    label_data.append(mean_d)
    return label_data



def get_data_dist_sim(model_name, split_name, dist_name, props, target_models):
    prop_dist_dict = load_av_similarities(model_name, split_name, dist_name)

    l_pos = ['all', 'all-some', 'some', 'few-some']
    l_neg = ['few']
    p_dicts = []
    for p in props:
        path = f'../data/aggregated_vocab/{p}.json'
        dist_dict = prop_dist_dict[p]
        with open(path) as infile:
            d = json.load(infile)
        n_pos = [c for c, d in d.items() if d['ml_label'] in l_pos and 
                 all([tm in d['model_vocabs'] for tm in target_models])] # and len(d['model_vocabs']) == 3]
        n_neg = [c for c, d in d.items() if d['ml_label'] in l_neg and
                all([tm in d['model_vocabs'] for tm in target_models])] #and len(d['model_vocabs']) == 3]
        prop_d = dict()
        if p.startswith('female-'):
            p = p.replace('female-', '')+'*'
        prop_d['property'] = p
        #prop_d['n_pos'] = len(n_pos)
        #prop_d['n_neg'] = len(n_neg)
        prop_d['total'] = len(n_pos)+len(n_neg)
        prop_d['pos'] = float(dist_dict['sim_pos'])
        #prop_d['sim_neg'] = float(dist_dict['sim_neg'])
        prop_d['pos_neg'] = float(dist_dict['sim_pos_neg'])
        prop_d['d'] = prop_d['pos'] - prop_d['pos_neg']
        p_dicts.append(prop_d)
    return p_dicts

def get_props(set_type = 'prop_set'):
    path_dir = '../data/aggregated_vocab/'
    files = os.listdir(path_dir)
    if set_type == 'prop_set':
        props = [p.split('.')[0] for p in files if not p.startswith('female')]
    elif set_type == 'ceiling':
        props = [p.split('.')[0] for p in files if p.startswith('female')]
    else:
        props = []
    props = [p for p in props if p != '']
    return props

def load_av_similarities(model_name, split_name, dist_name):
    
    path = f'../analysis/similarities/{dist_name}/{model_name}/{split_name}.csv'
    
    with open(path) as infile:
        data = list(csv.DictReader(infile))
    
    prop_dist_dict = dict()
    
    for d in data:
        prop = d['property']
        prop_dist_dict[prop] = d
    return prop_dist_dict

def get_data_dist_size(props, model_names):

    l_pos = ['all', 'all-some', 'some', 'few-some']
    l_neg = ['few']
    p_dicts = []
    for p in props:
        path = f'../data/aggregated_vocab/{p}.json'
        #dist_dict = prop_dist_dict[p]
        with open(path) as infile:
            d = json.load(infile)
        n_pos = [c for c, d in d.items() if d['ml_label'] in l_pos ]
        n_neg = [c for c, d in d.items() if d['ml_label'] in l_neg ]
        n_pos_vocab = [c for c, d in d.items() if d['ml_label'] in l_pos and 
                       all([mn in d['model_vocabs'] for mn in model_names])]
        n_neg_vocab = [c for c, d in d.items() if d['ml_label'] in l_neg and 
                       all([mn in d['model_vocabs'] for mn in model_names])]
        prop_d = dict()
        if p.startswith('female-'):
            p = p.replace('female-', '')+'*'
        prop_d['property'] = p
        prop_d['n_pos'] = len(n_pos)
        prop_d['n_neg'] = len(n_neg)
        prop_d['total'] = len(n_pos)+len(n_neg)
        prop_d['n_pos_vocab'] = len(n_pos_vocab)
        prop_d['n_neg_vocab'] = len(n_neg_vocab)
        prop_d['total_vocab'] = len(n_pos_vocab)+len(n_neg_vocab)
        p_dicts.append(prop_d)
    return p_dicts




