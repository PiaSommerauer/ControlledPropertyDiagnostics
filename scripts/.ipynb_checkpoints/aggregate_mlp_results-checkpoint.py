import csv
import os
import json
from collections import defaultdict
import pandas as pd
import sys


def load_output(prop, split_name, model_dist, model_name, 
                clf_name, param_name, syn_control, pl_extension):
    
  
    path_dir = f'../output/{split_name}-{model_dist}/{model_name}/{clf_name}-{param_name}/{prop}/'
 
    path_file = f'{path_dir}test-syn_control_{syn_control}-pl_extension_{pl_extension}.csv'
    with open(path_file) as infile:
        data = list(csv.DictReader(infile))
    return data


def aggregate_mlp_results(split_name, model_name, model_dist, 
                          clf_name, param_name, prop, syn_control, pl_extension):
    cl_output = defaultdict(list)
    labels = dict()
    for n in range(10):
        param_name_updated = f'{param_name}-{n}'
        output = load_output(prop, split_name, model_dist, model_name, 
                               clf_name, param_name_updated, syn_control, pl_extension)
        for d in output:
            c = d['word']
            p = d['prediction']
            if n == 0:
                labels[c]= d['label']
            cl_output[c].append(p)

    # aggregate:
    d_aggregated = []

    for c, predictions in cl_output.items():
        c_pos = predictions.count('pos')
        c_neg = predictions.count('neg')
        if c_pos > c_neg:
            p_agg = 'pos'
        elif c_neg > c_pos:
            p_agg = 'neg'
        # if both labels were chosen equally often:
        else:
            p_agg = 'tie'
        d = dict()
        d['word'] = c
        d['label'] = labels[c]
        d['prediction'] = p_agg
        d_aggregated.append(d)
    return d_aggregated
        
    
    
def aggregated_results_to_file(d_aggregated, split_name, model_name, model_dist, 
                          clf_name, param_name, prop, syn_control, pl_extension):
    
    
    path_dir = f'../output/{split_name}-{model_dist}/{model_name}/{clf_name}-{param_name}-aggregated/{prop}'
    print(path_dir)
    os.makedirs(path_dir, exist_ok = True)
    path_file = f'{path_dir}/test-syn_control_{syn_control}-pl_extension_{pl_extension}.csv'
    header = d_aggregated[0].keys()
    with open(path_file, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames = header)
        writer.writeheader()
        for d in d_aggregated:
            writer.writerow(d)
    
    
def get_props():
    path_dir = '../data/aggregated_vocab/'
    files = os.listdir(path_dir)
    props = [p.split('.')[0] for p in files]
    return props


def main():
    pl_extension = False
    syn_control = False
    split_name = 'standard-cosine'
    model_name = sys.argv[1]
    model_dist = sys.argv[2]
    #model_dist = 'giga-google-wiki'


    clf_params = [('mlp', '50--300'), ('mlp', '50-50-300'), 
                    ('mlp', '100--300'), ('mlp', '100-100-300')]


    props = get_props()
    props = [p for p in props if p != '']

    for prop in props:

        for clf_name, param_name in clf_params:
            d_aggregated = aggregate_mlp_results(split_name, model_name, model_dist, 
                          clf_name, param_name, prop, syn_control, pl_extension)

            aggregated_results_to_file(d_aggregated, split_name, model_name, model_dist, 
                          clf_name, param_name, prop, syn_control, pl_extension)
            
if __name__ == '__main__':
    main()