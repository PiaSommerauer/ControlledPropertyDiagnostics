import csv
import numpy as np
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import os

import utils_comparison
import pandas as pd

import csv
import os
import json
from collections import defaultdict
from collections import Counter
import pandas as pd


def load_output(prop, split_name, model_dist, model_name, 
                clf_name, param_name, syn_control, pl_extension):
    
    if not model_dist is None:
        path_dir = f'../output/{split_name}-{model_dist}/{model_name}/{clf_name}-{param_name}/{prop}/'

    path_file = f'{path_dir}test-syn_control_{syn_control}-pl_extension_{pl_extension}.csv'
    
    with open(path_file) as infile:
        data = list(csv.DictReader(infile))
    return data


# def load_data(model_dist, model_name, pl_extension, syn_control):
#     path = f'../evaluation/overviews/{model_dist}-{model_name}-{syn_control}-{pl_extension}.csv'
#     #with open(path) as infile:
#      #   data = list(csv.DictReader(infile))
#     df = pd.read_csv(path)
#     data = df.to_dict('records')
#     return data


def load_prop_set(prop):
    path = f'../data/aggregated/{prop}.json'
    with open(path) as infile:
        prop_dict = json.load(infile)
    return prop_dict




def load_train(model_dist, prop):
    
    path = f'../data/train_test_splits/standard-cosine/{model_dist}/{prop}/train-syn_control_False-pl_extension_False.csv'
    if os.path.isfile(path):
        with open(path) as infile:
            data = list(csv.DictReader(infile))
    else:
        data = []
    return data


def get_vector_data(data, model):
    
    matrix = []
    labels = []
    word_index_dict = dict()
    i = 0
    words = []
    for d in data:
        word = d['word']
        label = d['label']
        if word in model.vocab:
            hyp_dict = dict()
            vec = model[word]
            matrix.append(vec)
            labels.append(label)
            word_index_dict[word] = i
            words.append(word)
            i += 1
    matrix = np.array(matrix)
    return words, matrix, labels, word_index_dict

def get_props(set_type = 'standard'):
    path_dir = '../data/aggregated/'
    files = os.listdir(path_dir)
    if set_type == 'standard':
        props = [p.split('.')[0] for p in files if not p.startswith('female')]
    elif set_type == 'ceiling':
        props = [p.split('.')[0] for p in files if p.startswith('female')]
    props = [p for p in props if p != '']
    return props


def get_train_test_data(prop, split_name, model_dist, model_name, model, clf_name, param_name):
    data = []
    syn_control = False
    pl_extension = False

    output = load_output(prop, split_name, model_dist, model_name, 
                    clf_name, param_name, syn_control, pl_extension)
    
    train = load_train(model_dist, prop)
    if train != []:

        words_train, vecs_train, labels_train, wi_dict_train = get_vector_data(train, model)
        words_out, vecs_out, labels_out, wi_dict_out = get_vector_data(output, model)
        word_pred_dict = dict()
        for d in output:
            word_pred_dict[d['word']] = d['prediction']

        
        for w_out, i_out in wi_dict_out.items():
            vec_out = vecs_out[i_out]
            dists = pairwise_distances(vecs_train, [vec_out], metric = 'cosine')
            indices_sorted = np.argsort(dists.flatten())
            top_i = indices_sorted[0]
            w_train = words_train[top_i]
            dist = dists[top_i]
            l_train = labels_train[top_i]
            l_out = labels_out[i_out]
            pred = word_pred_dict[w_out]
            if l_out == pred:
                result = 'correct'
            else:
                result = 'incorrect'
            if l_train == l_out:
                equ = True
            else:
                equ = False
            d = dict()
            d['property'] = prop
            d['word_train'] = w_train
            d['word_test'] = w_out
            d['label_train'] = l_train
            d['label_test'] = l_out
            d['prediction'] = pred
            d['equivalent'] = equ
            d['outcome'] = result
            d['dist'] = dist[0]
            data.append(d)
    return data

def get_prop_clf_mapping(data):
    
    mapping = dict()
    to_clean = {"'", '(', ')'}
    for d in data:
        prop = d['property']
        top_select_clf = d['top_select']
        for char in to_clean:
            top_select_clf = top_select_clf.replace(char, '')
        clf_name, param_name = top_select_clf.split(', ')
        param_name = param_name.replace('mean', 'aggregated')
        mapping[prop] = (clf_name, param_name)
    return mapping

def get_examples(split_name, model_dist, model_dict, props):
    syn_control = False
    pl_extension = False
    equ = [True, False]
    res = ['correct', 'incorrect']
    model_name = model_dict['name']
    model = model_dict['model']
    #props_learned = model_dict['props'].split(' ')
    data_performance = utils_comparison.load_data(model_dist, model_name, pl_extension, syn_control)
    prop_clf_mapping = get_prop_clf_mapping(data_performance)
    prop_outcome = dict()
    for prop, (clf_name, param_name) in prop_clf_mapping.items():
        if prop.startswith('ceiling-'):
            prop = prop.replace('ceiling-', 'female-')
            #print('modified prop', prop)
        if prop in props:
            #print('found', prop)
            prop_outcome[prop] = get_train_test_data(prop, split_name, model_dist, model_name, model, clf_name, param_name)
    return prop_outcome

def get_prop_overview(split_name, model_dist, model_dict, props):
    prop_overview = []
    prop_outcome = get_examples(split_name, model_dist, model_dict, props)
    for prop, outcome in prop_outcome.items():
        equ_correct = len([d for d in outcome if d['equivalent']==True and d['outcome'] == 'correct'])
        equ_incorrect = len([d for d in outcome if d['equivalent']==True and d['outcome'] == 'incorrect'])
        nequ_correct = len([d for d in outcome if d['equivalent']==False and d['outcome'] == 'correct'])
        nequ_incorrect = len([d for d in outcome if d['equivalent']==False and d['outcome'] == 'incorrect'])
        d = dict()
        total_same = equ_correct + equ_incorrect
        total_diff = nequ_correct + nequ_incorrect
        if total_same != 0 and total_diff != 0:
            #if 'female-' in prop:
             #   prop = prop.replace('female-', '')+'*'
            d['property'] = prop
            #d[('same', 'correct')] = equ_correct/total_same
            #d[('same', 'incorrect')] = equ_incorrect/total_same
            #d['total'] = total_same + total_diff
            d[('diff', 'total')] = total_diff
            d[('diff', 'correct-abs')] = nequ_correct
            d[('diff', 'correct-norm')] = nequ_correct /total_diff
            d[('same', 'total')] = total_same
            d[('same', 'correct-abs')] = equ_correct
            d[('same', 'correct-norm')] = equ_correct/total_same
            #d[('diff', 'incorrect')] = nequ_incorrect/total_diff
            prop_overview.append(d)
    return prop_overview, prop_outcome