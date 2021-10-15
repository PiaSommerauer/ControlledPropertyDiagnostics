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


def load_data(model_dist, model_name, pl_extension, syn_control):
    path = f'../evaluation/overviews/{model_dist}-{model_name}-{syn_control}-{pl_extension}.csv'
    #with open(path) as infile:
     #   data = list(csv.DictReader(infile))
    df = pd.read_csv(path)
    data = df.to_dict('records')
    return data


def load_prop_set(prop):
    path = f'../data/aggregated/{prop}.json'
    with open(path) as infile:
        prop_dict = json.load(infile)
    return prop_dict


def get_concept_relation_dict(prop_dict):
    c_dict = defaultdict(dict)
    for c, d in prop_dict.items():
        rel_d = d['relations']
        for rel, v in rel_d.items():
            if v > 0.5:
                c_dict[c][rel] = v
    return c_dict

def get_props():
    path_dir = '../data/aggregated/'
    files = os.listdir(path_dir)
    props = [p.split('.')[0] for p in files]
    return props


