# TODO start code for exploration

from collections import defaultdict
import glob
import csv
import numpy as np
from nltk.corpus import wordnet as wn
import inflect
import random
from sklearn.metrics import pairwise_distances


def sort_by_key(data_dict_list, keys):
    """

    :param data_dict_list: table data (list of dicts)
    :param keys: keys to sort data by
    :return: dict sorting data by keys
    """
    sorted_dict = defaultdict(list)
    for d in data_dict_list:
        if len(keys) == 1:
            key = keys[0]
            sortkey = d[key]
            if type(sortkey) == str:
                sortkey = sortkey.strip()
        else:
            sortkeys = []
            for key in keys:
                sortkey = d[key].strip()
                sortkeys.append(sortkey)
            sortkey = tuple(sortkeys)
        sorted_dict[sortkey].append(d)
    return sorted_dict


def load_lexical_data():
    data_by_concept = defaultdict(list)
    paths = glob.glob('../data/vocabulary_data/vocab_by_property/*.csv')
    for path in paths:
        with open(path) as infile:
            data_dicts = list(csv.DictReader(infile))
            for d in data_dicts:
                concept = d['concept']
                data_by_concept[concept].append(d)
    return data_by_concept


def load_data(run, group, source):
    path_dir = f'../data/clean/annotations_{source}/'
    path_files = f'{path_dir}run{run}-group_{group}/*.csv'
    data = []
    for f in glob.glob(path_files):
        with open(f) as infile:
            data.extend(list(csv.DictReader(infile)))
    return data


def load_crowd_truth(data_iterations, source):
    # e.g. source = clean_contradictions_batch_0.5
    runs = [i[1] for i in data_iterations]
    experiment = 'all'

    quid_uas_dict = dict()

    # ct is always calculated on entire set
    path_dir = '../data/crowd_truth/results/'
    path_f = f'{path_dir}run{"_".join(runs)}-group_-all--batch-all--{source}-units.csv'
    # run3_4_5_pilot_5_scalar_heat-group_experiment-all--batch-all--data_processed-annotations.csv
    with open(path_f) as infile:
        dict_list = csv.DictReader(infile)
        for d in dict_list:
            pos_resp = d['unit_annotation_score_true']
            quid = d['unit']
            quid_uas_dict[quid] = float(pos_resp)
    return quid_uas_dict


def get_av(pr_list):
    if len(pr_list) > 0:
        av = sum(pr_list) / len(pr_list)
    else:
        av = 0
    return av


def get_synonym_pairs(concepts):
    synonyms = dict()
    concept_syns = defaultdict(set)

    for c in concepts:
        syns = wn.synsets(c, 'n')
        concept_syns[c].update(syns)

    for c1, syns1 in concept_syns.items():
        for c2, syns2 in concept_syns.items():
            if c1 != c2:
                # check if there is synset overlap
                overlap = syns1.intersection(syns2)
                if overlap:
                    synonym_pair = tuple(sorted([c1, c2]))
                    if synonym_pair not in synonyms:
                        d = dict()
                        d['shared'] = overlap
                        d['#total'] = len(syns1.union(syns2))
                        for c, syn_sets in zip([c1, c2], [syns1, syns2]):
                            d[c] = syn_sets
                        synonyms[synonym_pair] = d
    return synonyms


def get_centroid(positive_examples, model, pluralize=False):
    engine = inflect.engine()
    matrix = []
    oov = []
    for w in positive_examples:
        # get pl:

        if w in model.vocab:
            vec = model[w]
            matrix.append(vec)
        else:
            oov.append(w)

    matrix = np.array(matrix)
    cent = np.mean(matrix, axis=0)
    return cent, oov


def get_distances_to_centroid(centroid, all_concepts, model):
    engine = inflect.engine()
    distance_concept_list = []
    oov = []
    for w in all_concepts:
        if w in model.vocab:
            vec = model[w]
            cosine = np.dot(centroid, vec) / (np.linalg.norm(centroid) * np.linalg.norm(vec))
            distance_concept_list.append((cosine, w, w))
        # else:
        #   plural = engine.plural(w)
        #  if plural in model.vocab:
        #     vec = model[plural]
        #    cosine = np.dot(centroid, vec)/(np.linalg.norm(centroid)*np.linalg.norm(vec))
        #   distance_concept_list.append((cosine, plural, w))
        else:
            oov.append(w)

    return distance_concept_list, oov


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


def get_closest_word(target_word, words, word_index_dict, matrix):
    i_target = word_index_dict[target_word]
    vec_target = matrix[i_target]
    dists = pairwise_distances(matrix, [vec_target], metric='cosine')
    indices_sorted = np.argsort(dists.flatten())

    return indices_sorted


def search_random_with_seed(model, all_concepts, n_pos):
    matrix, words, oov = get_matrix(all_concepts, model)
    word_index_dict = dict()
    for n, w in enumerate(words):
        word_index_dict[w] = n

    # pick seed that is in model vocab
    seed = random.sample(words, 1)
    target_word = seed[0]
    words_selected = set()
    words_selected.add(target_word)

    search_words = True
    while search_words:
        indices_sorted = get_closest_word(target_word, words, word_index_dict, matrix)
        for i in indices_sorted:
            w = words[i]
            if w not in words_selected:
                words_selected.add(w)
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

    random_pos = words_selected
    # use words in model
    random_neg = [w for w in words if w not in random_pos]
    return seed[0], random_pos, random_neg


