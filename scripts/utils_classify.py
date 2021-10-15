from sklearn.metrics import pairwise_distances

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import os
import csv
import numpy as np
import pandas as pd

# my own model
#from KNCentroid import KNCentroid


def load_set(prop, model_dist_name, split_name, data_split, syn_control=False, pl_extension=False):
    path_dir = f'../data/train_test_splits/{split_name}/{model_dist_name}/{prop}/'
    path_f = f'{path_dir}{data_split}-syn_control_{syn_control}-pl_extension_{pl_extension}.csv'
    with open(path_f) as infile:
        data = list(csv.DictReader(infile))
    return data


def get_properties(model_dist_name, split_name):
    props = set()

    path = f'../data/train_test_splits/{split_name}/{model_dist_name}/'
    for d in os.listdir(path):
        props.add(d)
    props = [p for p in props if p != '.DS_Store']
    return props


def get_vector_data(data, model):
    matrix = []
    labels = []
    hyp_dicts = []
    word_index_dict = dict()
    i = 0
    for d in data:
        word = d['word']
        label = d['label']
        if word in model.vocab:
            hyp_dict = dict()
            vec = model[word]
            matrix.append(vec)
            labels.append(label)
            hyp_dict['word'] = word
            hyp_dict['hyp'] = d['hyp']
            hyp_dict['pos_rate'] = d['hyp_rate']
            hyp_dict['top_rel_rep'] = d['hyp_rel']
            hyp_dict['cosine_centroid'] = d['cosine_centroid']
            hyp_dict['label'] = label
            hyp_dicts.append(hyp_dict)
            word_index_dict[word] = i
            i = + 1
    matrix = np.array(matrix)
    return matrix, labels, word_index_dict, hyp_dicts


def get_baseline_f1(labels_test):
    baseline_dict = dict()
    total = len(labels_test)
    n_pos = labels_test.count('pos')
    n_neg = total - n_pos

    p_pos = n_pos / total
    p_neg = n_neg / total

    if p_pos > p_neg:
        p_maj = p_pos
        maj_label = 'pos'
    else:
        p_maj = p_neg
        maj_label = 'neg'

    pred_always_true = [maj_label for n in range(total)]
    p, r, f1, supp = precision_recall_fscore_support(labels_test, pred_always_true,
                                                     average='weighted', zero_division=0)

    # baseline_dict['f1_majority'] = p_maj
    baseline_dict['f1_majority'] = f1

    return baseline_dict


def classify(clf, matrix_train, matrix_test, labels_train, labels_test):
    res_dict = dict()
    clf.fit(matrix_train, labels_train)
    predictions = clf.predict(matrix_test)
    p, r, f1, supp = precision_recall_fscore_support(labels_test, predictions,
                                                     average='weighted', zero_division=0)
    acc = accuracy_score(labels_test, predictions)
    res_dict['p'] = p
    res_dict['r'] = r
    res_dict['f1_weighted'] = f1
    res_dict['acc'] = acc
    res_dict['n_pos'] = list(labels_test).count('pos')
    res_dict['predictions_pos'] = list(predictions).count('pos')
    res_dict['n_examples'] = len(labels_test)
    return res_dict, predictions


def classify_random_inits(clf, data_train, data_test, n=10):
    res_dicts = []
    labels_test = [d['label'] for d in data_test]
    labels_train = [d['label'] for d in data_train]

    n_pos_test = list(labels_test).count('pos')
    n_neg_test = len(labels_test) - n_pos_test

    n_pos_train = list(labels_train).count('pos')
    n_neg_train = len(labels_train) - n_pos_train

    labels_test = ['pos' for i in range(n_pos_test)]
    [labels_test.append('neg') for i in range(n_neg_test)]

    labels_train = ['pos' for i in range(n_pos_train)]
    [labels_train.append('neg') for i in range(n_neg_train)]

    for i in range(n):
        res_dict = dict()
        matrix_train = np.random.rand(len(labels_train), 300)
        matrix_test = np.random.rand(len(labels_test), 300)
        clf.fit(matrix_train, labels_train)
        predictions = clf.predict(matrix_test)
        p, r, f1, supp = precision_recall_fscore_support(labels_test, predictions,
                                                         average='weighted', zero_division=0)
        res_dict['p'] = p
        res_dict['r'] = r
        res_dict['f1_weighted'] = f1
        res_dict['n_pos'] = list(labels_test).count('pos')
        res_dict['predictions_pos'] = list(predictions).count('pos')
        res_dict['n_examples'] = len(labels_test)
        res_dicts.append(res_dict)
    mean_res_dict = dict()
    res_keys = res_dicts[0].keys()
    for k in res_keys:
        mean = sum([d[k] for d in res_dicts]) / len(res_dicts)
        mean_res_dict[k] = mean
    return mean_res_dict


def results_to_file(prop, syn_control, pl_extension, split_name, model_dist_name, model_name,
                    clf_name, predictions, hyp_dicts_test, param_name):
    path_dir = f'../output/{split_name}-{model_dist_name}/{model_name}/{clf_name}-{param_name}/{prop}/'
    os.makedirs(path_dir, exist_ok=True)
    path_file = f'{path_dir}test-syn_control_{syn_control}-pl_extension_{pl_extension}.csv'
    fieldnames = list(hyp_dicts_test[0].keys())
    fieldnames.append('prediction')
    with open(path_file, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for p, hyp_d in zip(predictions, hyp_dicts_test):
            hyp_d['prediction'] = p
            writer.writerow(hyp_d)


def classify_set(model_name, model_dist_name, model, split_name, clf_name, props, pl_extension, syn_control):
    print('\n classify set\n')
    clf_dict = {'lr': LogisticRegression, 'mlp': MLPClassifier}
                #'kn': KNeighborsClassifier, 'kncentroid': KNCentroid}
    clf_param_dict = {
                        'lr': [{'random_state': 0}],
                        'mlp': [{'hidden_layer_sizes': (50,), 'max_iter': 300},
                              {'hidden_layer_sizes': (50, 50), 'max_iter': 300},
                              {'hidden_layer_sizes': (100,), 'max_iter': 300},
                               {'hidden_layer_sizes': (100, 100), 'max_iter': 300}]
                      }
                      # 'kn': [{'n_neighbors': 1}, {'n_neighbors': 2},
                      #        {'n_neighbors': 3}, {'n_neighbors': 4},
                      #        {'n_neighbors': 5}, {'n_neighbors': 6},
                      #        {'n_neighbors': 7}, {'n_neighbors': 8}],
                      # 'kncentroid': [{'model': model}]}

    clf_params = clf_param_dict[clf_name]

    # add 10 runs to mlp
    if clf_name == 'mlp':
        params_extended = []
        for param_dict in clf_params:
            for n in range(10):
                new_param_dict = dict()
                new_param_dict.update(param_dict)
                new_param_dict['run'] = n
                params_extended.append(new_param_dict)
        clf_params = params_extended
    param_df_dict = dict()
    print('number parameters', len(clf_params))
    for params in clf_params:
        param_name = '-'.join([str(p) for n, p in params.items() if n != 'model'])
        to_replace = {'(': '', ')': '', ',': '-', ' ': ''}
        for char1, char2 in to_replace.items():
            param_name = param_name.replace(char1, char2)
        # remove 'run' from param dict going into clf object:
        if clf_name == 'mlp':
            params.pop('run')
        results_dicts = []
        for prop in props:
            data_split = 'train'
            data_train = load_set(prop, model_dist_name, split_name, data_split,
                                  syn_control=syn_control, pl_extension=pl_extension)
            data_split = 'test'
            data_test = load_set(prop, model_dist_name,
                                 split_name, data_split, pl_extension=pl_extension)
            m_train, l_train, word_index_dict, hyp_dicts_train = get_vector_data(data_train, model)
            m_test, l_test, word_index_dict_test, hyp_dicts_test = get_vector_data(data_test, model)
            words_test = [d['word'] for d in hyp_dicts_test]
            baseline_dict = get_baseline_f1(l_test)
            clf = clf_dict[clf_name]
            clf = clf(**params)
            prop_dict = dict()
            prop_dict['property'] = prop
            res_dict, predictions = classify(clf, m_train, m_test, l_train, l_test)
            res_dict.update(baseline_dict)
            results_to_file(prop, syn_control, pl_extension, split_name, model_dist_name, model_name,
                            clf_name, predictions, hyp_dicts_test, param_name)

            prop_dict.update(res_dict)
            results_dicts.append(prop_dict)

        df = pd.DataFrame(results_dicts)
        param_df_dict[param_name] = df
        path_dir = f'../evaluation/{split_name}-{model_dist_name}/{model_name}/{clf_name}-{param_name}/'
        os.makedirs(path_dir, exist_ok=True)
        path_file = f'{path_dir}test-syn_control_{syn_control}-pl_extension_{pl_extension}.csv'
        df.to_csv(path_file)
    return param_df_dict


def classify_random_vectors(model_name, model_dist_name, model, split_name,
                            clf_name, props, pl_extension, syn_control):
    print('\n classify random vecs\n')
    clf_dict = {'lr': LogisticRegression, 'mlp': MLPClassifier}
               # 'kn': KNeighborsClassifier, 'kncentroid': KNCentroid}
    clf_param_dict = {'lr': [{'random_state': 0}],
                      'mlp': [{'hidden_layer_sizes': (50,), 'max_iter': 300},
                              {'hidden_layer_sizes': (50, 50), 'max_iter': 300},
                              {'hidden_layer_sizes': (100,), 'max_iter': 300},
                              {'hidden_layer_sizes': (100, 100), 'max_iter': 300}]}
                      # 'kn': [{'n_neighbors': 1}, {'n_neighbors': 2},
                      #        {'n_neighbors': 3}, {'n_neighbors': 4},
                      #        {'n_neighbors': 5}, {'n_neighbors': 6},
                      #        {'n_neighbors': 7}, {'n_neighbors': 8}],
                      # 'kncentroid': [{'model': model}]}

    clf_params = clf_param_dict[clf_name]

    # add 100 runs to mlp
    if clf_name == 'mlp':
        print('updating param name for mlp')
        params_extended = []
        for param_dict in clf_params:
            for n in range(10):
                new_param_dict = dict()
                new_param_dict.update(param_dict)
                new_param_dict['run'] = n
                params_extended.append(new_param_dict)
        clf_params = params_extended
    param_df_dict = dict()
    split_name_random = 'random_vectors'
    print('number parameters', len(clf_params))
    for params in clf_params:
        param_name = '-'.join([str(p) for n, p in params.items() if n != 'model'])
        to_replace = {'(': '', ')': '', ',': '-', ' ': ''}
        for char1, char2 in to_replace.items():
            param_name = param_name.replace(char1, char2)
        # remove 'run' from param dict going into clf object:
        if clf_name == 'mlp':
            print(params.keys())
            params.pop('run')
        results_dicts = []
        for prop in props:
            data_split = 'train'
            data_train = load_set(prop, model_dist_name, split_name, data_split,
                                  syn_control=syn_control, pl_extension=pl_extension)
            data_split = 'test'
            data_test = load_set(prop, model_dist_name,
                                 split_name, data_split, pl_extension=pl_extension)
            l_test = [d['label'] for d in data_test]
            baseline_dict = get_baseline_f1(l_test)

            clf = clf_dict[clf_name]
            clf = clf(**params)
            prop_dict = dict()
            prop_dict['property'] = prop
            res_dict = classify_random_inits(clf, data_train, data_test, n=10)
            res_dict.update(baseline_dict)

            prop_dict.update(res_dict)
            results_dicts.append(prop_dict)

        df = pd.DataFrame(results_dicts)
        param_df_dict[param_name] = df
        path_dir = f'../evaluation/{split_name_random}/{model_name}/{clf_name}-{param_name}/'
        os.makedirs(path_dir, exist_ok=True)
        path_file = f'{path_dir}test-syn_control_{syn_control}-pl_extension_{pl_extension}.csv'
        df.to_csv(path_file)
    return param_df_dict

