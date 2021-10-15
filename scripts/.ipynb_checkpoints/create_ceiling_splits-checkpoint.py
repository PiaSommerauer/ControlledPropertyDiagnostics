import csv
import utils
import json
import random
import os

# set random seed - done after running experiments  for the first time
random.seed(3)



def load_prop_set(prop):
    path = f'../data/aggregated_vocab/{prop}.json'
    with open(path) as infile:
        prop_dict = json.load(infile)
    return prop_dict


def get_distribution(prop_dict, target_models):
    label_dict = dict()
    label_dict['pos'] = ['all', 'all-some', 'some', 'few-some']
    label_dict['neg'] = ['few']
    label_dict_inv = dict()
    for ml_label, labels in label_dict.items():
        for l in labels:
            label_dict_inv[l] = ml_label
    distribution_dict = dict()
    for c, d in prop_dict.items():
        label = d['ml_label']
        model_vocabs = d['model_vocabs']
        model_vocabs = d['model_vocabs']
        
        check_vocabs = [mn in model_vocabs for mn in target_models]
      
        # check if legitimate label and concept in all model vocabs
        if label in label_dict_inv and all(check_vocabs):
            label_bin = label_dict_inv[label]
            if label_bin not in distribution_dict:
                distribution_dict[label_bin] = 1
            else:
                distribution_dict[label_bin] += 1
    return distribution_dict


def create_gender_distribution(prop_dict_female, prop_dict_target, target_models):
    labels_pos = ['all', 'all-some', 'some', 'few-some']
    labels_neg = ['few']

    # load label distribution - function checks if concepts in all model vocabs
    dist_female = get_distribution(prop_dict_female, target_models)
    dist_prop_target = get_distribution(prop_dict_target, target_models)
    print(dist_female)
    print(dist_prop_target)
    enough_data = []
    for l in ['pos', 'neg']:
        female = dist_female[l]
        target = dist_prop_target[l]
        if female >= target:
            enough_data.append(True)
        else:
            enough_data.append(False)
    print(enough_data)
    if not all(enough_data):
        print('not enough control data')
        subsampled_dict = None
    else:
        print('enough data')

        control_pos = [(c, d) for c, d in prop_dict_female.items() if d['ml_label'] in labels_pos]
        control_neg = [(c, d) for c, d in prop_dict_female.items() if d['ml_label'] in labels_neg]
        print(len(control_pos), len(control_neg))
        target_pos = dist_prop_target['pos']
        control_pos_random_selection = random.sample(control_pos, target_pos)
        print(len(control_pos_random_selection))
        target_neg = dist_prop_target['neg']
        control_neg_random_selection = random.sample(control_neg, target_neg)
        print(len(control_neg_random_selection))
        subsampled_dict = dict()
        subsampled_dict.update(control_pos_random_selection)
        subsampled_dict.update(control_neg_random_selection)
        print(len(subsampled_dict))
    return subsampled_dict


def subsampled_dict_to_file(target_prop, subsampled_dict):
    target_path = f'../data/aggregated_vocab/female-{target_prop}.json'
    with open(target_path, 'w') as outfile:
        json.dump(subsampled_dict, outfile, indent=4)


def get_props():
    path_dir = '../data/aggregated_vocab/'
    files = os.listdir(path_dir)
    props = [p.split('.')[0] for p in files if not p.startswith('female')]
    props = [p for p in props if p != '']
    return props


def main():
    
    #target_models = {'wiki_full', 'googlenews', 'giga'}
    target_models = {'giga_corpus', 'wiki_corpus', 'googlenews'}

    target_props =  get_props()
    prop = 'female'
    prop_dict_female = load_prop_set(prop)
    prop_dict_female_in_vocab = dict()
    # remove concepts not in all models:
    for concept, d in prop_dict_female.items():
        model_vocabs = d['model_vocabs']
        check_vocabs = [mn in model_vocabs for mn in target_models]
        if all(check_vocabs):
            prop_dict_female_in_vocab[concept] = d

    for target_prop in target_props:
        print(target_prop)
        prop_dict_target = load_prop_set(target_prop)
        prop_dict_target_in_vocab = dict()
        # remove concepts not in all models:
        for concept, d in prop_dict_target.items():
            model_vocabs = d['model_vocabs']
            check_vocabs = [mn in model_vocabs for mn in target_models]
            if all(check_vocabs):
                prop_dict_target_in_vocab[concept] = d

        subsampled_dict = create_gender_distribution(prop_dict_female_in_vocab, prop_dict_target_in_vocab, target_models)
        if subsampled_dict is not None:
            subsampled_dict_to_file(target_prop, subsampled_dict)


if __name__ == '__main__':
    main()