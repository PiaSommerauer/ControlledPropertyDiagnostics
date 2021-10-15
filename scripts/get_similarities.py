import pandas as pd
from gensim.models import KeyedVectors
import utils_control
import os
import sys


def get_similarities_sets(properties, split_name, model_dist_name, model):
    sims_dicts = []
    for prop in properties:
        sim_dict = dict()
        sim_dict['property'] = prop
        sim_dict['split_name'] = split_name
        if split_name in ['random-words-seed', 'random-words-no-dist']:
            sims_pos = []
            sims_neg = []
            sims_pos_neg = []
            for n in range(10):
                split_name_updated = split_name + f'-{n}'
                data_split = 'test'
                prop_set = utils_control.load_set(prop, model_dist_name, split_name_updated, data_split)
                data_split = 'train'
                prop_set.extend(utils_control.load_set(prop, model_dist_name, split_name_updated, data_split))
                pos = [d['word'] for d in prop_set if d['label'] == 'pos']
                neg = [d['word'] for d in prop_set if d['label'] == 'neg']
                pairs_pos = get_pairs(pos)
                pairs_pos_neg = get_pairs(pos, neg)
                pairs_neg = get_pairs(neg)
                sims_pos.append(get_cosines(pairs_pos, model))
                sims_neg.append(get_cosines(pairs_neg, model))
                sims_pos_neg.append(get_cosines(pairs_pos_neg, model))
            sim_dict['sim_pos'] = sum(sims_pos) / len(sims_pos)
            sim_dict['sim_neg'] = sum(sims_neg) / len(sims_neg)
            sim_dict['sim_pos_neg'] = sum(sims_pos_neg) / len(sims_pos_neg)
        else:

            data_split = 'test'
            prop_set = utils_control.load_set(prop, model_dist_name, split_name, data_split)
            data_split = 'train'
            prop_set.extend(utils_control.load_set(prop, model_dist_name, split_name, data_split))
            pos = [d['word'] for d in prop_set if d['label'] == 'pos']
            neg = [d['word'] for d in prop_set if d['label'] == 'neg']
            pairs_pos = get_pairs(pos)
            pairs_pos_neg = get_pairs(pos, neg)
            pairs_neg = get_pairs(neg)
            sim_dict['sim_pos'] = get_cosines(pairs_pos, model)
            sim_dict['sim_neg'] = get_cosines(pairs_neg, model)
            sim_dict['sim_pos_neg'] = get_cosines(pairs_pos_neg, model)
        sims_dicts.append(sim_dict)
    return sims_dicts


def get_pairs(l, l2=None):
    pairs = []
    if l2 is None:
        l2 = l
    for i1 in l:
        for i2 in l2:
            if i1 != i2:
                p = {i1, i2}
                pairs.append(p)
    return pairs


def get_cosines(pairs, model):
    cosines = []
    for w1, w2 in pairs:
        if all([w in model.vocab for w in [w1, w2]]):
            cosines.append(model.similarity(w1, w2))
    mean = sum(cosines) / len(cosines)
    return mean


def main():
    pl_extension = False
    syn_control = False
    model_dist_name = 'giga-google-wiki'

    model_name = sys.argv[1]
    model_path = sys.argv[2]
    model_dist_name = sys.argv[3]

    # model_name = 'googlennews'
    # model_path = 'model_path = '../../../../Data/dsm/word2vec/GoogleNews-vectors-negative300.bin'

    # model_name = 'wiki_full'
    # model_path = '../../../../Data/dsm/wikipedia_full/sgns_pinit1/sgns_pinit1/sgns_rand_pinit1.words'

    # model_name = 'giga'
    # model_path = '../../../../Data/dsm/nlpl_models/12/model.bin'

    if model_path.endswith('.bin'):
        bin_encoding = True
    else:
        bin_encoding = False
    model = KeyedVectors.load_word2vec_format(model_path, binary=bin_encoding)
    model_dict = dict()
    model_dict['model'] = model
    model_dict['name'] = model_name

    #model_dist_name = 'giga-google-wiki'


    properties = utils_control.get_props()
    properties = [p for p in properties if p != '' and not p.startswith('female-')]

    # # random seed
    # split_name = 'random-words-seed'
    # sims_dicts = get_similarities_sets(properties, split_name, model_dist_name, model)
    # dir_name = f'../analysis/similarities/{model_dist_name}/{model_name}/'
    # os.makedirs(dir_name, exist_ok=True)
    # path = f'{dir_name}{split_name}.csv'
    # df = pd.DataFrame(sims_dicts)
    # df.to_csv(path)
    #
    # # random no dist
    # split_name = 'random-words-no-dist'
    # sims_dicts = get_similarities_sets(properties, split_name, model_dist_name, model)
    # dir_name = f'../analysis/similarities/{model_dist_name}/{model_name}/'
    # os.makedirs(dir_name, exist_ok=True)
    # path = f'{dir_name}{split_name}.csv'
    # df = pd.DataFrame(sims_dicts)
    # df.to_csv(path)

    # standard distribution
    split_name = 'standard-cosine'
    sims_dicts = get_similarities_sets(properties, split_name, model_dist_name, model)
    dir_name = f'../analysis/similarities/{model_dist_name}/{model_name}/'
    os.makedirs(dir_name, exist_ok=True)
    path = f'{dir_name}{split_name}.csv'
    df = pd.DataFrame(sims_dicts)
    df.to_csv(path)


if __name__ == '__main__':
    main()