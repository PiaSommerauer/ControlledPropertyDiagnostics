import pandas as pd
import utils_comparison

import os
import sys

def mean_mlp_to_files(model_name, model_dist, syn_control, pl_extension):
    # get mean mlp performances
    clf_params = [('mlp', '50--300'), ('mlp', '50-50-300'),
                  ('mlp', '100--300'), ('mlp', '100-100-300')]
    for clf_name, param_name in clf_params:
        utils_comparison.mean_mlp(model_name, model_dist, clf_name, param_name,
                                  pl_extension=pl_extension, syn_control=syn_control)

def main():

    model_name = sys.argv[1]
    model_dist = sys.argv[2]

    # wikipedia
    # parameters:
    pl_extension = False
    syn_control = False
    #model_dist = 'giga-google-wiki'


    mean_mlp_to_files(model_name, model_dist, syn_control, pl_extension)
    split_res_dict = utils_comparison.get_comparison_splits(model_name,
                                                            model_dist,
                                                            add_nn=False,
                                                            add_random_seed=True,
                                                            add_random_words_no_dist=True)

    # print(split_res_dict.keys())
    data_selectivity, cols = utils_comparison.get_selectivity(split_res_dict, add_nn=False)
    df = pd.DataFrame(data_selectivity)
    dir_path = '../evaluation/overviews'
    os.makedirs(dir_path, exist_ok=True)
    path = f'{dir_path}/{model_dist}-{model_name}-{syn_control}-{pl_extension}.csv'
    df = df[cols].set_index('property')
    df.to_csv(path)

if __name__ == '__main__':
    main()