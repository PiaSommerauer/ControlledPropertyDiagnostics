from collections import defaultdict
import pandas as pd
import os
import numpy as np

import json
from collections import Counter



def load_results(model_name, model_dist, split_name, clf_name, param_name, pl_extension=False, syn_control=False):
    if not model_dist is None:
        path_dir = f'../evaluation/{split_name}-{model_dist}/{model_name}/{clf_name}-{param_name}/'
    else:
        path_dir = f'../evaluation/{split_name}/{model_name}/{clf_name}-{param_name}/'
    path_file = f'{path_dir}test-syn_control_{syn_control}-pl_extension_{pl_extension}.csv'
    # with open(path_file) as infile:
    #   data = list(csv.DictReader(infile))
    df = pd.read_csv(path_file, index_col='property')
    cols = df.columns
    col_unnamed = cols[0]
    df = df.drop(col_unnamed, axis=1)
    props_set = []
    control = []
    for p, row in df.iterrows():
        d = row.to_dict()
        if p.startswith('female-'):
            d['property'] = p.split('-')[1]
            control.append(d)
        else:
            d['property'] = p
            props_set.append(d)

    df_control = pd.DataFrame(control).set_index('property')
    df_set = pd.DataFrame(props_set).set_index('property')

    return df_set, df_control


def get_comparison_splits(model_name, model_dist, add_random_seed=False, add_nn=False,
                          add_random_words_no_dist=False, pl_extension=False, syn_control=False):
    split_names = ['standard-cosine', 'random_vectors']
    control_names = ['random_vectors', 'majority']

    ######

    if add_random_seed == True:
        random_seed_names = [f'random-words-seed-{n}' for n in range(10)]
        split_names.extend(random_seed_names)
        control_names.extend(random_seed_names)
    if add_random_words_no_dist == True:
        random_no_dist = [f'random-words-no-dist-{n}' for n in range(10)]
        split_names.extend(random_no_dist)
        control_names.extend(random_no_dist)

    probe_param_names = [('lr', '0'),
                         ('mlp', '50--300-mean'), ('mlp', '50-50-300-mean'),
                         ('mlp', '100--300-mean'), ('mlp', '100-100-300-mean')]
    if add_nn:
        nn_param_names = [('kn', '1'), ('kn', '2'), ('kn', '3'), ('kncentroid', '')]
        clf_param_names = probe_param_names + nn_param_names
    else:
        clf_param_names = probe_param_names

    split_res_dict = defaultdict(dict)
    for split_name in split_names:
        if split_name == 'random_vectors':
            model_dist_updated = None
        else:
            model_dist_updated = model_dist
        for clf_name, param_name in clf_param_names:
            df_set, df_control = load_results(model_name, model_dist_updated, split_name,
                                              clf_name, param_name,
                                              pl_extension=pl_extension, syn_control=syn_control)
            df = df_set.sort_values('f1_weighted', ascending=False)
            for prop, f1 in df['f1_weighted'].items():
                split_res_dict[(clf_name, param_name, split_name)][prop] = f1
                f1_col = df_control['f1_weighted']
                if prop in f1_col.keys():
                    control_f1 = f1_col[prop]
                    split_res_dict[(clf_name, param_name, split_name)][f'ceiling-{prop}'] = control_f1

    # take distribution-based baselines from last df
    for prop, row in df.iterrows():
        majority = row['f1_majority']
        split_res_dict['majority'][prop] = majority
        # split_res_dict['f1_always_pos'][prop] = row['f1_always_pos']

    for prop, row in df_control.iterrows():
        prop = f'ceiling-{prop}'
        majority = row['f1_majority']
        split_res_dict['majority'][prop] = majority
        # split_res_dict['f1_always_pos'][prop] = row['f1_always_pos']

    return split_res_dict


def get_selectivity(split_res_dict, add_nn=False):
    df_all = pd.DataFrame(split_res_dict)
    properties = list(df_all.index)
    dict_list = df_all.to_dict('records')

    # print(dict_list[0])

    # split_names = ['standard-cosine', 'random-words', 'random_vectors']
    control_names = ['random_vectors', 'majority']
    random_seeds = [f'random-word-seed-{n}' for n in range(10)]
    random_no_dist = [f'random-words-no-dist-{n}' for n in range(10)]
    probe_param_names = [('lr', '0'),
                         ('mlp', '50--300-mean'), ('mlp', '50-50-300-mean'),
                         ('mlp', '100--300-mean'), ('mlp', '100-100-300-mean')]
    if add_nn:
        nn_param_names = [('kn', '1'), ('kn', '2'), ('kn', '3'), ('kncentroid', '')]
        clf_param_names = probe_param_names + nn_param_names
    else:
        clf_param_names = probe_param_names

    updated_dicts = []
    for prop, d in zip(properties, dict_list):
        new_d = dict()
        new_d['property'] = prop
        # get top diagnostic classifier
        score_clf_dict = defaultdict(list)
        for clf in probe_param_names:
            clf_split_name = (clf[0], clf[1], 'standard-cosine')
            score = d[clf_split_name]
            score_clf_dict[score].append(clf)
        score_max = max(list(score_clf_dict.keys()))
        clf_max = score_clf_dict[score_max][0]
        # new_d['top_f1'] = score_max
        new_d['top_clfs'] = clf_max
        new_d['top_clfs_f1'] = score_max
        for clf_name_param in clf_param_names:
            new_d[clf_name_param] = d[(clf_name_param[0], clf_name_param[1], 'standard-cosine')]

        # get corresponding random baseline
        select_clf_dict = defaultdict(list)
        select_clf_dict_rand = defaultdict(list)
        for clf_name_param in clf_param_names:
            # get score
            clf_split_name = (clf_name_param[0], clf_name_param[1], 'standard-cosine')
            score = d[clf_split_name]

            clf_names_control = [(clf_name_param[0], clf_name_param[1], f'random-words-seed-{n}')
                                 for n in range(10)]
            clf_names_random_no_dist = [(clf_name_param[0], clf_name_param[1], f'random-words-no-dist-{n}')
                                        for n in range(10)]
            clf_scores_control = [d[clf_name_control] for clf_name_control in clf_names_control]
            clf_scores_random_no_dist = [d[clf_name_rand] for clf_name_rand in clf_names_random_no_dist]

            control_mean = sum(clf_scores_control) / len(clf_scores_control)
            control_max = max(clf_scores_control)
            control_min = min(clf_scores_control)

            rand_mean = sum(clf_scores_random_no_dist) / len(clf_scores_random_no_dist)
            rand_max = max(clf_scores_random_no_dist)
            rand_min = min(clf_scores_random_no_dist)

            # clf_name_rand_words = (clf_name_param[0], clf_name_param[1], 'random-words')
            # clf_score_rand_words = d[clf_name_rand_words]
            clf_name_rand_vecs = (clf_name_param[0], clf_name_param[1], 'random_vectors')
            clf_score_rand_vecs = d[clf_name_rand_vecs]

            # selectivity
            select = score - control_mean
            select_rand = score - rand_mean

            if clf_name_param in probe_param_names:
                select_clf_dict[select].append(clf_name_param)
                select_clf_dict_rand[select_rand].append(clf_name_param)
            # score_control = d[clf_name_control]
            # select = score_max - score_control
            new_d[f'select-seed-mean-{clf_name_param}'] = select
            new_d[f'random-seed-mean-{clf_name_param}'] = control_mean
            new_d[f'random-seed-max-{clf_name_param}'] = control_max
            new_d[f'random-seed-min-{clf_name_param}'] = control_min

            new_d[f'select-rand-no-dist-mean-{clf_name_param}'] = select_rand
            new_d[f'rand-no-dist-mean-{clf_name_param}'] = rand_mean
            new_d[f'rand-no-dist-max-{clf_name_param}'] = rand_max
            new_d[f'rand-no-dist-min-{clf_name_param}'] = rand_min

            # new_d[f'random-words-{clf_name_param}'] = clf_score_rand_words
            new_d[f'random-vecs-{clf_name_param}'] = clf_score_rand_vecs

        top_select = max(list(select_clf_dict.keys()))
        top_select_rand = max(list(select_clf_dict_rand.keys()))
        top_clf_select = select_clf_dict[top_select][0]
        top_clf_select_rand = select_clf_dict_rand[top_select_rand][0]
        new_d['top_select'] = top_clf_select
        new_d['top_select_f1'] = top_select
        new_d['top_select_rand'] = top_clf_select_rand
        new_d['top_select_rand_f1'] = top_select_rand
        # add majority baseline
        clf_name_maj = 'majority'
        new_d[f'control-{clf_name_maj}'] = d[clf_name_maj]
        updated_dicts.append(new_d)

    # sorted cols
    cols = []
    cols.append('top_clfs')
    cols.append('top_clfs_f1')
    cols.append('top_select')
    cols.append('top_select_f1')
    current_cols = [k for k in updated_dicts[0].keys()]
    seed_cols = [c for c in current_cols if str(c).startswith('select-seed-mean-')]
    rand_cols = [c for c in current_cols if str(c).startswith('select-rand-no-dist-')]
    for clf_name_param in probe_param_names:
        cols.append(clf_name_param)
    for c in seed_cols:
        cols.append(c)
    for c in rand_cols:
        cols.append(c)
    # add remeining cols
    for c in current_cols:
        if c not in cols:
            cols.append(c)

    return updated_dicts, cols


def get_mean_df(dfs):
    prop_dicts = defaultdict(list)

    for df in dfs:
        props = df.index

        dicts_set = df.to_dict('records')
        for prop, d in zip(props, dicts_set):
            prop_dicts[prop].append(d)

    keys = prop_dicts[props[0]][0].keys()
    prop_mean_dict = dict()
    for prop, data in prop_dicts.items():
        mean_d = dict()
        mean_d['property'] = prop
        for k in keys:
            values = []
            for d in data:
                v = d[k]
                values.append(v)
            m = sum(values) / len(values)
            mean_d[k] = m
        prop_mean_dict[prop] = mean_d
    df = pd.DataFrame(prop_mean_dict)
    return df.T


def mean_mlp(model_name, model_dist, clf_name, param_name,
             pl_extension=False, syn_control=False):
    split_names = ['standard-cosine', 'random_vectors']
    random_seed_names = [f'random-words-seed-{n}' for n in range(10)]
    random_words_no_dist = [f'random-words-no-dist-{n}' for n in range(10)]
    split_names.extend(random_seed_names)
    split_names.extend(random_words_no_dist)

    for split_name in split_names:
        if split_name == 'random_vectors':
            model_dist_updated = None
            split_model_dist_name = split_name
        else:
            model_dist_updated = model_dist
            split_model_dist_name = f'{split_name}-{model_dist}'
        dfs = []
        dfs_control = []
        for n in range(10):
            param_name_updated = param_name + f'-{n}'
            df_set, df_control = load_results(model_name, model_dist_updated, split_name,
                                              clf_name, param_name_updated,
                                              pl_extension=pl_extension, syn_control=syn_control)
            dfs.append(df_set)
            dfs_control.append(df_control)
            # dfs_set.append(df_set)
            # dfs_control.append(df_control)
        mean_df = get_mean_df(dfs)
        mean_df_control = get_mean_df(dfs_control)

        mean_dicts = mean_df.to_dict('record')
        mean_dicts_control = mean_df_control.to_dict('record')

        for d in mean_dicts_control:
            prop = d['property']
            d['property'] = f'female-{prop}'
            mean_dicts.append(d)

        df = pd.DataFrame(mean_dicts)

        path_dir = f'../evaluation/{split_model_dist_name}/{model_name}/{clf_name}-{param_name}-mean/'
        os.makedirs(path_dir, exist_ok=True)
        path_file = f'{path_dir}test-syn_control_{syn_control}-pl_extension_{pl_extension}.csv'
        df.to_csv(path_file)


def simplify_headers(d):
    new_d = dict()
    clf_dist_dict = dict()
    dist_short_dict = {'standard-cosine': 'p', 'random-words': 'rw', 'random_vectors': 'rv'}
    mlp_dict = {'50--300-mean': '1', '50-50-300-mean': '2', '100--300-mean': '3', '100-100-300-mean': '4'}
    for k, v in d.items():
        if type(k) == tuple:
            clf = k[0]
            param = k[1]
            dist = k[2]
            if clf == 'mlp':
                clf = clf + mlp_dict[param]
            dist = dist_short_dict[dist]
            new_d[f'{clf}-{dist}'] = v
        else:
            new_d[k] = v
    keys_sorted = sorted(list(new_d.keys()))
    keys_sorted.pop(keys_sorted.index('property'))
    keys_sorted.pop(keys_sorted.index('majority'))
    keys_sorted.insert(0, 'property')
    keys_sorted.insert(1, 'majority')
    return new_d, keys_sorted


def get_below_random(split_res_dict):
    df_all = pd.DataFrame(split_res_dict)
    properties = list(df_all.index)
    dict_list = df_all.to_dict('records')

    controls = ['majority', 'random-words', 'random_vectors', ]
    prop_max_dict = dict()
    prop_data_dict = dict()
    prop_ceiling_dict = dict()
    for prop, d in zip(properties, dict_list):
        v_probe_dict = defaultdict(list)
        for k, v in d.items():
            v_probe_dict[v].append(k)
        top_v = max(list(v_probe_dict.keys()))
        top_probes = v_probe_dict[top_v]
        prop_max_dict[prop] = top_probes
        if prop.startswith('ceiling-'):
            prop_ceiling_dict[prop.split('-')[1]] = d
        else:
            prop_data_dict[prop] = d

    excluded = dict()
    for prop, max_perf in prop_max_dict.items():
        for perf in max_perf:
            for c in controls:
                if c in perf:
                    excluded[prop] = max_perf
                    break

    data_excluded = []
    for prop in excluded:
        d = prop_data_dict[prop]
        d_ceiling = prop_ceiling_dict[prop]
        d['property'] = prop
        d_ceiling['property'] = prop + '-ceiling'
        d, h_sorted = simplify_headers(d)
        d_ceiling, h_sorted = simplify_headers(d_ceiling)
        data_excluded.extend([d, d_ceiling])
    return data_excluded, h_sorted


def get_max_select(d, selectivities):
    values = []
    for s in selectivities:
        values.append(d[s])
    max_v = max(values)
    if max_v > 0:
        select = True
    else:
        select = False
    return select, max_v


def get_learned_not_learned(data_selectivity):
    prop_data = [d for d in data_selectivity if not d['property'].startswith('ceiling')]
    ceiling_data = [d for d in data_selectivity if d['property'].startswith('ceiling')]
    ceiling_dict = dict()
    prop_data_dict = dict()
    for d in ceiling_data:
        prop = d['property']
        prop_data_dict[prop] = d
        prop_only = prop.split('-')[1]
        ceiling_dict[prop_only] = d

    selectivities = ["select-seed-mean-('lr', '0')",
                     "select-seed-mean-('mlp', '50--300-mean')",
                     "select-seed-mean-('mlp', '50-50-300-mean')",
                     "select-seed-mean-('mlp', '100--300-mean')",
                     "select-seed-mean-('mlp', '100-100-300-mean')"]

    props_select = set()
    props_not_select = set()
    props_max_selectivity = dict()
    for d in prop_data:
        select, max_v = get_max_select(d, selectivities)
        props_max_selectivity[d['property']] = max_v
        if select:
            props_select.add(d['property'])
        else:
            props_not_select.add(d['property'])
    return props_select, props_not_select, prop_data_dict


def load_data(model_dist, model_name, pl_extension, syn_control):
    path = f'../evaluation/overviews/{model_dist}-{model_name}-{syn_control}-{pl_extension}.csv'
    # with open(path) as infile:
    #   data = list(csv.DictReader(infile))
    df = pd.read_csv(path)
    data = df.to_dict('records')
    return data


def get_control_overview(data, control, target_clf=None, set_type='prop_set'):
    controls = [control]  # 'random-vecs']
    # clfs = [target_clf]

    if not target_clf is None:
        controls = [f"{c}-('{target_clf[0]}', '{target_clf[1]}')" for c in controls]

    # controls.append('control-majority')

    control_data = []
    controls_updated = set()
    for d in data:
        new_d = dict()
        prop = d['property']
        # print(prop)
        if set_type == 'prop_set':
            if not prop.startswith('ceiling-'):
                new_d['property'] = prop
                for k, v in d.items():
                    if target_clf is None:
                        target_control = [k.startswith(control) for control in controls]
                    else:
                        target_control = [k == control for control in controls]
                    if any(target_control):
                        new_d[k] = float(v)
                        controls_updated.add(k)
                control_data.append(new_d)
        elif set_type == 'ceiling_set':
            if prop.startswith('ceiling-'):
                new_d['property'] = prop
                for k, v in d.items():
                    if target_clf is None:
                        target_control = [k.startswith(control) for control in controls]
                    else:
                        target_control = [k == control for control in controls]
                    if any(target_control):
                        new_d[k] = float(v)
                        controls_updated.add(k)
                control_data.append(new_d)
    print()
    # get_mean
    new_d = dict()
    new_d['property'] = 'mean'
    # print(controls_updated)
    for control in controls_updated:
        values = [d[control] for d in control_data if not np.isnan(d[control])]
        # print(values)
        mean = sum(values) / len(values)
        # print(control, mean)
        new_d[control] = mean
    control_data.append(new_d)

    header = []
    header.append('property')
    for control_name in controls:
        targets = sorted([c for c in controls_updated if c.startswith(control_name)])
        header.extend(targets)

    return control_data, header


# def get_short_selectivity(data_selectivity):
#     selectivities = ["select-seed-mean-('lr', '0')",
#                          "select-seed-mean-('mlp', '50--300-mean')",
#                          "select-seed-mean-('mlp', '50-50-300-mean')",
#                          "select-seed-mean-('mlp', '100--300-mean')",
#                          "select-seed-mean-('mlp', '100-100-300-mean')"]
#     clfs_to_simple = {
#         ('lr', '0'): 'lr',
#         ('mlp', '50--300-mean'): 'mlp1',
#         ('mlp', '50-50-300-mean'): 'mlp2',
#         ('mlp', '100--300-mean'): 'mlp3',
#         ('mlp', '100-100-300-mean'): 'mlp4',
#     }
#     ceiling_dict = dict()
#     for d in data_selectivity:
#         p = d['property']
#         if p.startswith('ceiling-'):
#             p = p.split('-')[1]
#             ceiling_dict[p] = d

#     select_simple = ['lr', 'mlp1', 'mlp2', 'mlp3', 'mlp4']

#     select_to_simple = dict()
#     for sel_long, sel_short in zip(selectivities, select_simple):
#         select_to_simple[sel_long] = sel_short

#     # get selectivity only
#     data_selectivity_only = []
#     for d in data_selectivity:
#         new_d = dict()
#         p = d['property']
#         if p.startswith('ceiling-'):
#             p = p.split('-')[1]+'*'
#         new_d['property'] = p
#         top_clf = d['top_clfs']
#         new_d['top_clf_f1'] = clfs_to_simple[top_clf]
#         top_f1 = d['top_clfs_f1']
#         new_d['top_f1'] = top_f1
#         top_select = d['top_select']
#         new_d['top_probe_select'] = clfs_to_simple[top_select]
#         new_d['top_probe_f1'] = d['top_select_f1']

#         selects = []
#         f1s = []
#         for k, v in d.items():
#             if k in selectivities:
#                 selects.append(v)
#             if k in clfs_to_simple.keys():
#                 f1s.append(v)
#         st_dev_select = stdev(selects)
#         st_dev_f1 = stdev(f1s)
#         new_d['stdev_f1'] = st_dev_f1
#         new_d['stdev_select'] = st_dev_select

#         data_selectivity_only.append(new_d)
#     return data_selectivity_only


def get_selectivity_data(data):
    clf_dict = {
        "('lr', '0')": 'lr',
        "('mlp', '50--300-mean')": 'mlp1',
        "('mlp', '50-50-300-mean')": 'mlp2',
        "('mlp', '100--300-mean')": 'mlp1',
        "('mlp', '100-100-300-mean')": 'mlp2',
    }

    selectivity_data = []

    ceiling_data = defaultdict(dict)
    prop_data = dict()
    for d in data:
        prop = d['property']
        # simplify clfs:/

        for k, v in d.items():
            if v in clf_dict:
                d[k] = clf_dict[v]
        if prop.startswith('ceiling-'):
            prop = prop.split('-')[1]
            ceiling_data[prop] = d
        else:
            prop_data[prop] = d

    target_keys = ['property', 'top_clfs', 'top_clfs_f1', 'top_select', 'top_select_f1']  # , 'control-majority']
    for prop, d in prop_data.items():
        if prop != 'female':
            new_d = dict()
            # if prop in ceiling_data:
            d_ceiling = ceiling_data[prop]
            for k in target_keys:
                new_d[k] = d[k]
            if 'top_select' in d_ceiling:
                # print('found', d_ceiling['top_select'])
                new_d['ceiling_top_select'] = d_ceiling['top_select']
            else:
                new_d['ceiling_top_select'] = None
            if 'top_select_f1' in d_ceiling:
                new_d['ceiling_top_select_f1'] = d_ceiling['top_select_f1']
            else:
                new_d['ceiling_top_select_f1'] = None
            selectivity_data.append(new_d)
    return selectivity_data


def get_mean(rel_counts, n_concepts):
    for rel, v in rel_counts.items():
        n_options = n_concepts[rel]
        rel_counts[rel] = v / n_options


def remove_neg(rel_counts):
    rel_pos = ['variability_limited', 'typical_of_concept',
               'implied_category', 'typical_of_property',
               'affording_activity', 'afforded_usual',
               'variability_open', 'afforded_unusual', True, False]
    clean_counts = Counter()
    for k, v in rel_counts.items():
        if k in rel_pos:
            clean_counts[k] = v
    return clean_counts


def load_prop_set(prop):
    path = f'../data/aggregated/{prop}.json'
    with open(path) as infile:
        prop_dict = json.load(infile)
    return prop_dict


def get_rel_agg(prop, data_splits, relations='all', cnt='pos_response_rate'):
    split_name = 'standard-cosine'
    model_dist_name = 'giga-google-wiki'

    rel_counts = Counter()
    rel_counts_total = Counter()
    example_count = 0

    full_set = load_prop_set(prop)
    for c, d in full_set.items():
        # d = full_set[c]
        rel_d = d['relations']
        l = d['ml_label']
        if l not in [None, 'few']:
            example_count += 1
            # count how many times a relation occurred as an option
            if relations == 'hypothesis':
                # check if there are positive options:
                if 'typical_of_concept' in rel_d:
                    rel_counts_total[True] += 1
                    rel_counts_total[False] += 1
                else:
                    rel_counts_total[False] += 1
            # count how often each relation is available for normalization
            for rel in rel_d:
                rel_counts_total[rel] += 1

            # add up proportion of positive responses
            if relations == 'all':
                if cnt == 'pos_response_rate':
                    for rel, v in rel_d.items():
                        if v > 0.5:
                            rel_counts[rel] += v
                else:
                    for rel, v in rel_d.items():
                        if v > 0.5:
                            rel_counts[rel] += 1

            elif relations == 'top':
                rels = d['rel_hyp']
                prop_hyp = d['prop_hyp']
                for rel in rels:
                    if cnt == 'pos_response_rate':
                        rel_counts[rel] += prop_hyp
                    elif cnt == 'n_top':
                        rel_counts[rel] += 1
            elif relations == 'hypothesis':
                hyp = d['hypothesis']
                if cnt == 'n_top':
                    rel_counts[hyp] += 1
                elif cnt == 'pos_response_rate':
                    rel_counts[hyp] += d['prop_hyp']

    return rel_counts, rel_counts_total, example_count


def get_total_relation_responses(props_select, props_not_select, relations='top', cnt='pos_response_rate'):
    rel_counts_learned = Counter()
    rel_counts_not_learned = Counter()

    n_rels_learned = Counter()
    ex_learned = 0
    for prop in props_select:
        if prop != 'female':
            # print(prop)
            data_splits = ['test', 'train']
            rel_counts, total_rel_counts, ex_cnt = get_rel_agg(prop,
                                                               data_splits, relations=relations,
                                                               cnt=cnt)
            n_rels_learned += total_rel_counts
            rel_counts_learned += rel_counts
            ex_learned += ex_cnt

    n_rels_not_learned = Counter()
    ex_not_learned = 0
    for prop in props_not_select:
        if prop != 'female':
            # print(prop)
            data_splits = ['test', 'train']
            rel_counts, total_rel_counts, ex_cnt = get_rel_agg(prop, data_splits, relations=relations, cnt=cnt)
            n_rels_not_learned += total_rel_counts
            rel_counts_not_learned += rel_counts
            ex_not_learned += ex_cnt

    # n_relations = n_rels_learned + n_rels_not_learned
    rel_counts_total = rel_counts_learned + rel_counts_not_learned
    rel_counts_learned = remove_neg(rel_counts_learned)
    rel_counts_not_learned = remove_neg(rel_counts_not_learned)
    # normalize
    for r, cnt in rel_counts_learned.items():
        # rel_counts_learned[r] = cnt/n_rels_learned[r]
        rel_counts_learned[r] = cnt / rel_counts_total[r]
    for r, cnt in rel_counts_not_learned.items():
        # rel_counts_not_learned[r] = cnt/n_rels_not_learned[r]
        rel_counts_not_learned[r] = cnt / rel_counts_total[r]

    return rel_counts_learned, rel_counts_not_learned  # , n_relations


def get_table(rel_counts_learned, rel_counts_not_learned):
    rel = []
    all_relations = set()
    all_relations.update(rel_counts_learned.keys())
    all_relations.update(rel_counts_not_learned.keys())

    relations_repr = ['variability_limited',
                      'affording_activity', 'afforded_usual', 'typical_of_property']

    for relation in all_relations:
        d = dict()
        rels_learned = rel_counts_learned[relation]
        rels_not_learned = rel_counts_not_learned[relation]
        if relation in relations_repr:
            hyp = 'yes'
        else:
            hyp = 'no'
        # total = rels_learned + rels_not_learned
        d['relation'] = relation
        d['hyp.'] = hyp
        # d['learned_abs'] = rels_learned
        d['learned'] = rels_learned
        # d['not_learned_abs'] = rels_not_learned
        d['not_learned'] = rels_not_learned
        d['-'] = '-'
        # d['total'] = total
        rel.append(d)
    return rel


def get_mean(rel_counts, n_concepts):
    for rel, v in rel_counts.items():
        n_options = n_concepts[rel]
        rel_counts[rel] = v / n_options


def remove_neg(rel_counts):
    rel_pos = ['variability_limited', 'typical_of_concept',
               'implied_category', 'typical_of_property',
               'affording_activity', 'afforded_usual',
               'variability_open', 'afforded_unusual', True, False]
    clean_counts = Counter()
    for k, v in rel_counts.items():
        if k in rel_pos:
            clean_counts[k] = v
    return clean_counts
