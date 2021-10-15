import os
from gensim.models import KeyedVectors
import json

from DataSplit import DataSplit

import sys



def load_prop_set(prop):
    path = f'../data/aggregated_vocab/{prop}.json'
    with open(path) as infile:
        prop_dict = json.load(infile)
    return prop_dict

def get_props():
    path_dir = '../data/aggregated_vocab/'
    files = os.listdir(path_dir)
    props = [p.split('.')[0] for p in files]
    return props


def main():
    # google model
    model_path = sys.argv[1]
    test = sys.argv[2]
    model_name = sys.argv[3]
    if test == 'True':
        test = True
    else:
        test = False
    # e.g. '../../../../Data/dsm/word2vec/GoogleNews-vectors-negative300.bin'
    
    model_google = KeyedVectors.load_word2vec_format(model_path, binary=True)
    model = model_google
    
    # target_models for vocab check:
    #target_models = {'wiki_full', 'googlenews', 'giga'}
    target_models = {'giga_corpus', 'wiki_corpus', 'googlenews'}

    if test == True:
        model_name = model_name+'-TEST'
    else:
        model_name = model_name
    split_name = 'standard-cosine'
    split_name_random_seed = 'random-words-seed'
    split_name_random_words_no_dist = 'random-words-no-dist'
    seeds = range(10)
    n_random_no_dist = range(10)
    pl_extension = False
    synonymy_control = False

    if test == True:
        props = ['yellow']
    else:

        props = get_props()
        props = [p for p in props if p != '']
    print('Running test:', test)
    print(sorted(props))
    for prop in props:
        print()
        print(prop)
        print()
        prop_dict = load_prop_set(prop)
        # only use concepts in all three models#
        prop_dict_clean = dict()
        for concept, d in prop_dict.items():
            model_vocabs = d['model_vocabs']
            check_vocabs = [mn in model_vocabs for mn in target_models]
            if all(check_vocabs):
                prop_dict_clean[concept] = d
        split = DataSplit(prop, prop_dict_clean, model,
                          synonymy_control=synonymy_control, pl_extension=pl_extension, test_split_size=0.40)
        split.create_splits()
        split.data_to_file(model_name, split_name)
        # random structured
        for n in seeds:
            split.get_random_seed_splits()
            split_name_random_seed_n = f'{split_name_random_seed}-{n}'
            split.data_to_file(model_name, split_name_random_seed_n)
        # random random
        for n in n_random_no_dist:
            split.get_random_splits_no_dist()
            split_name_random_words_no_dist_n = f'{split_name_random_words_no_dist}-{n}'
            print(split_name_random_words_no_dist_n)
            split.data_to_file(model_name, split_name_random_words_no_dist_n)


if __name__ == '__main__':
    main()