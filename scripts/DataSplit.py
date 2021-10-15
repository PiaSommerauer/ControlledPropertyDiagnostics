import glob
import csv
import os
import utils
from collections import defaultdict

import inflect
import random


class DataSplit:
    def __init__(self, prop, prop_dict, model,
                 synonymy_control=False, pl_extension=True,
                 prior_neg=0.0, test_split_size=0.33):
        self.prop = prop
        self.prop_dict = prop_dict
        self.model = model
        self.synonymy_control = synonymy_control
        self.pl_extension = pl_extension
        self.prior_neg = prior_neg
        self.n_total = len(prop_dict)
        self.test_split_size = test_split_size

    def create_splits(self):

        self.concepts = [c for c, d in self.prop_dict.items()]
        self.concept_label_dict = self.get_labels()
        self.label_dict = dict()
        self.label_dict['pos'] = [c for c, l in self.concept_label_dict.items() if l == 'pos']
        self.label_dict['neg'] = [c for c, l in self.concept_label_dict.items() if l == 'neg']

        if self.synonymy_control:
            self.synonyms_by_word = self.get_synonym_mapping()

        self.concept_cosine_dict = self.get_cosines()
        self.concept_hyp_dict = self.get_hypothesis_info()

        self.test = defaultdict(list)
        self.train = dict()

        # prior_pos = self.prioritize_hyp('pos')
        # prior_neg = self.prioritize_hyp('neg')

        self.test['pos'] = []
        self.test['neg'] = []

        self.fill_test_set('pos')
        self.fill_test_set('neg')

        self.fill_train_set('pos')
        self.fill_train_set('neg')

        if self.pl_extension:
            self.sg_pl_mapping = self.get_sg_pl_mapping()
            self.train, self.test = self.extend_with_plurals('standard')
            # get cosines based on new set
            self.concept_cosine_dict = self.get_cosine_dict_extended()

    def get_random_splits(self):

        self.test_random = dict()
        self.train_random = dict()
        self.test_random['pos'] = []
        self.test_random['neg'] = []

        self.train_random['pos'] = []
        self.train_random['neg'] = []

        # print('test pos')
        self.get_random_test('pos')
        # print('test neg')
        self.get_random_test('neg')

        # print('train pos')
        self.get_random_train('pos')
        # print('train neg')
        self.get_random_train('neg')

        if self.pl_extension:
            self.sg_pl_mapping = self.get_sg_pl_mapping()
            self.train_random, self.test_random = self.extend_with_plurals('random')

    def get_random_splits_no_dist(self):

        self.concepts = [c for c, d in self.prop_dict.items()]
        self.concept_label_dict = self.get_labels()
        self.label_dict = dict()
        self.label_dict['pos'] = [c for c, l in self.concept_label_dict.items() if l == 'pos']
        self.label_dict['neg'] = [c for c, l in self.concept_label_dict.items() if l == 'neg']

        if self.pl_extension:
            self.sg_pl_mapping = self.get_sg_pl_mapping()
            self.train_random, self.test_random = self.extend_with_plurals('random')

        # all_concepts = [c for c in self.prop_dict.keys()]
        # limit to all_concepts in vocab:
        # all_concepts = [c for c in self.prop_dict.keys() if c in self.model.vocab]
        all_concepts = self.label_dict['pos'] + self.label_dict['neg']
        # limit to all_concepts in vocab:
        all_concepts = [c for c in all_concepts if c in self.model.vocab]

        n_pos = len([c for c in all_concepts if c in self.label_dict['pos']])
        print(f'pick {n_pos}  examples randomly.')

        self.concept_cosine_dict = self.get_cosines()
        self.concept_hyp_dict = self.get_hypothesis_info()

        # get random pos examples:
        concepts_pos = random.sample(all_concepts, n_pos)
        concepts_neg = [c for c in all_concepts if c not in concepts_pos]

        self.train_random_no_dist = dict()
        self.test_random_no_dist = dict()

        n_pos_test = int(round(len(concepts_pos) * self.test_split_size, 0))
        n_neg_test = int(round(len(concepts_neg) * self.test_split_size, 0))

        self.test_random_no_dist['pos'] = random.sample(concepts_pos, n_pos_test)
        self.test_random_no_dist['neg'] = random.sample(concepts_pos, n_neg_test)
        self.train_random_no_dist['pos'] = [c for c in concepts_pos
                                            if c not in self.test_random_no_dist['pos']]
        self.train_random_no_dist['neg'] = [c for c in concepts_pos
                                            if c not in self.test_random_no_dist['neg']]

        if self.pl_extension:
            self.sg_pl_mapping = self.get_sg_pl_mapping()
            self.train_random_no_dist, self.test_random_no_dist = self.extend_with_plurals('standard')
            # get cosines based on new set
            self.concept_cosine_dict = self.get_cosine_dict_extended()

    def get_random_seed_splits(self):

        self.concepts = [c for c in self.prop_dict.keys()]
        self.concept_label_dict = self.get_labels()
        self.label_dict = dict()
        self.label_dict['pos'] = [c for c, l in self.concept_label_dict.items() if l == 'pos']
        self.label_dict['neg'] = [c for c, l in self.concept_label_dict.items() if l == 'neg']

        if self.synonymy_control:
            self.synonyms_by_word = self.get_synonym_mapping()

        all_concepts = self.label_dict['pos'] + self.label_dict['neg']
        # limit to all_concepts in vocab:
        all_concepts = [c for c in all_concepts if c in self.model.vocab]
        n_pos = len([c for c in all_concepts if c in self.label_dict['pos']])
        print(f'searching for {n_pos} positive examples using a random seed.')

        self.concept_label_dict = self.get_labels()
        self.concept_cosine_dict = self.get_cosines()
        self.concept_hyp_dict = self.get_hypothesis_info()

        self.train_random_seed = dict()
        self.test_random_seed = dict()

        self.seed, random_pos, random_neg = utils.search_random_with_seed(self.model, all_concepts, n_pos)

        n_pos_test = int(round(len(random_pos) * self.test_split_size, 0))
        n_neg_test = int(round(len(random_neg) * self.test_split_size, 0))
        if n_pos_test < 1:
            n_pos_test = 1
            n_neg_test = n_neg_test - 1
        if n_neg_test < 1:
            n_neg_test = 1
            n_pos_test = n_pos_test - 1
        print(n_pos_test, n_neg_test)
        self.test_random_seed['pos'] = random.sample(random_pos, n_pos_test)
        self.test_random_seed['neg'] = random.sample(random_neg, n_neg_test)

        self.train_random_seed['pos'] = [c for c in random_pos if c not in self.test_random_seed['pos']]
        self.train_random_seed['neg'] = [c for c in random_neg if c not in self.test_random_seed['neg']]

        if self.pl_extension:
            self.sg_pl_mapping = self.get_sg_pl_mapping()
            self.train_random_seed, self.test_random_seed = self.extend_with_plurals('standard')
            # get cosines based on new set
            self.concept_cosine_dict = self.get_cosine_dict_extended()

    def load_splits(self, split_name, model_name, prop):

        self.concepts = [c for p, c in self.prop_set.pairs.keys()]
        self.concept_label_dict = self.get_labels()
        self.label_dict = dict()
        self.label_dict['pos'] = [c for c, l in self.concept_label_dict.items() if l == 'pos']
        self.label_dict['neg'] = [c for c, l in self.concept_label_dict.items() if l == 'neg']

        if self.synonymy_control:
            self.synonyms_by_word = self.get_synonym_mapping()

        self.concept_cosine_dict = self.get_cosines()
        self.concept_hyp_dict = self.get_hypothesis_info()

        self.test = dict()
        self.train = dict()

        if self.pl_extension:
            self.sg_pl_mapping = self.get_sg_pl_mapping()

        prop = self.prop
        dir_path = f'../data/train_test_splits/{split_name}/{model_name}/{prop}/'
        data_splits = ['test', 'train']
        all_data = []
        for ds in data_splits:
            path_split = f'{dir_path}{ds}-syn_control_{self.synonymy_control}-pl_extension_{self.pl_extension}.csv'
            with open(path_split) as infile:
                data = list(csv.DictReader(infile))
                all_data.extend(data)
            data_pos = [d for d in data if d['label'] == 'pos']
            data_neg = [d for d in data if d['label'] == 'neg']
            split_dict = getattr(self, ds)
            split_dict['pos'] = [c for c in data_pos]
            split_dict['neg'] = [c for c in data_neg]

    def get_sg_pl_mapping(self):
        engine = inflect.engine()
        sg_pl_mapping = dict()
        for c in self.prop_dict.keys():
            plural = engine.plural(c)
            sg_pl_mapping[c] = plural
            sg_pl_mapping[plural] = c
        return sg_pl_mapping

    def get_hypothesis_info(self):
        concept_hyp_dict = dict()
        for c, d in self.prop_dict.items():
            d_new = dict()
            d_new['hyp'] = d['hypothesis']
            d_new['hyp_rel'] = d['rel_hyp']
            d_new['hyp_rate'] = d['prop_hyp']
            concept_hyp_dict[c] = d_new
        return concept_hyp_dict

    def get_labels(self):
        concept_label_dict = dict()
        label_dict = dict()
        label_dict['pos'] = ['all', 'all-some', 'some', 'few-some']
        label_dict['neg'] = ['few']
        label_dict_inv = dict()
        for ml_label, labels in label_dict.items():
            for l in labels:
                label_dict_inv[l] = ml_label

        for c, d in self.prop_dict.items():
            l = d['ml_label']
            if l is not None:
                ml_label = label_dict_inv[l]
                concept_label_dict[c] = ml_label

        return concept_label_dict

    def get_cosines(self):
        concept_cosine_dict = dict()
        centroid, oov = utils.get_centroid(self.label_dict['pos'], self.model)
        distance_concept_list, oov = utils.get_distances_to_centroid(centroid, self.concepts, self.model)
        for cosine, wf, concept in distance_concept_list:
            concept_cosine_dict[concept] = cosine
        for c in oov:
            concept_cosine_dict[concept] = 'oov'
        return concept_cosine_dict

    def get_synonym_mapping(self):
        synonyms = utils.get_synonym_pairs(self.concepts)
        synonyms_by_word = defaultdict(set)
        for w1, w2 in synonyms.keys():
            synonyms_by_word[w1].add(w2)
            synonyms_by_word[w2].add(w1)
        return synonyms_by_word

    def prioritize_hyp(self, label):
        concepts_test = []
        concepts = self.label_dict[label]
        n_prioritize = round(len(concepts) * self.prior_neg, 0)
        print('number of prioritized examples shoud be:', n_prioritize)
        for c in concepts:
            hyp_dict = self.concept_hyp_dict[c]
            hyp = hyp_dict['hyp']
            prop_true = hyp_dict['hyp_rate']
            # prioritize examples expected not to be represented:
            # impose higher threshold to ensure high quality
            if (hyp == False) and prop_true > 0.70:
                concepts_test.append(c)
            if len(concepts_test) >= n_prioritize:
                break
        return concepts_test

    def fill_test_set(self, label):
        n_test = round(len(self.label_dict[label]) * self.test_split_size)
        print(n_test, len(self.label_dict[label]))
        # determine number of examples to select to fulfil prop
        n_remaining = n_test - len(self.test[label])
        print('remaining number to fill', n_remaining)
        prop_remaining = n_remaining / len(self.label_dict[label])

        # sort concepts in set by distance to centroid
        concepts_by_distance = []
        no_cos = []
        for c in self.label_dict[label]:
            if c in self.concept_cosine_dict:
                cos = self.concept_cosine_dict[c]
                if cos != 'oov':
                    # check if concept not already included because it was prioritized
                    if c not in self.test[label]:
                        concepts_by_distance.append((cos, c))

        # Calculate step size for sampline
        step_size = round(1 / prop_remaining, 0)
        if self.synonymy_control == False:
            for n, (cos, c) in enumerate(sorted(concepts_by_distance)):
                if n % step_size == 0:
                    self.test[label].append(c)
                if len(self.test[label]) >= n_test:
                    break
        else:
            for n, (cos, c) in enumerate(sorted(concepts_by_distance)):
                if n % step_size == 0:
                    self.test[label].append(c)
                    syns = self.synonyms_by_word[c]
                    for s in syns:
                        self.test[label].append(s)
                if len(self.test[label]) >= n_test:
                    break

    def fill_train_set(self, label):
        self.train[label] = []
        for c in self.label_dict[label]:
            if c not in self.test[label]:
                self.train[label].append(c)

    def get_random_test(self, label):

        n_test = self.test_split_size * len(self.label_dict[label])
        # use all concepts
        all_concepts = self.label_dict['pos'] + self.label_dict['neg']
        # exlcude already selected concepts:
        comp_label_dict = {'pos': 'neg', 'neg': 'pos'}
        comp_label = comp_label_dict[label]
        available_concepts = [c for c in all_concepts if c not in self.test_random[comp_label]]
        prop_test = n_test / len(available_concepts)
        # print('looking for a random test set of size', prop_test)
        # sort concepts in set by distance to centroid
        concepts_by_distance = []
        no_cos = []
        for c in available_concepts:
            if c in self.concept_cosine_dict:
                cos = self.concept_cosine_dict[c]
                if cos != 'oov':
                    # check if concept not already included because it was prioritized
                    if c not in self.test[label]:
                        concepts_by_distance.append((cos, c))

        # Calculate step size for sampline
        step_size = round(1 / prop_test, 0)
        # print('selecting concepts from sorted cosine list with step size:', step_size)
        if self.synonymy_control == False:
            for n, (cos, c) in enumerate(sorted(concepts_by_distance)):
                if n % step_size == 0:
                    self.test_random[label].append(c)
                if len(self.test_random[label]) >= n_test:
                    break
        else:
            for n, (cos, c) in enumerate(sorted(concepts_by_distance)):
                if n % step_size == 0:
                    self.test_random[label].append(c)
                    syns = self.synonyms_by_word[c]
                    for s in syns:
                        self.test_random[label].append(s)
                if len(self.test_random[label]) >= n_test:
                    break

    def get_random_train(self, label):

        n_train = len(self.label_dict[label]) - len(self.test_random[label])
        comp_label_dict = {'pos': 'neg', 'neg': 'pos'}
        comp_label = comp_label_dict[label]

        # use all concepts
        all_concepts = self.label_dict['pos'] + self.label_dict['neg']
        # exlcude already selected concepts:
        concepts_test = self.test_random['pos'] + self.test_random['neg']
        available_concepts = [c for c in all_concepts if c not in concepts_test]
        prop_train = n_train / len(available_concepts)
        # sort concepts in set by distance to centroid
        concepts_by_distance = []
        no_cos = []
        for c in available_concepts:
            if c in self.concept_cosine_dict:
                cos = self.concept_cosine_dict[c]
                if cos != 'oov':
                    # check if concept not already included because it was prioritized
                    if c not in self.test[label]:
                        concepts_by_distance.append((cos, c))

        # Calculate step size for sampline
        step_size = round(1 / prop_train, 0)
        if self.synonymy_control == False:
            for n, (cos, c) in enumerate(sorted(concepts_by_distance)):
                if (len(available_concepts) < step_size) or (step_size < 1):
                    self.train_random[label].append(c)
                elif n % step_size == 0:
                    self.train_random[label].append(c)
                if len(self.train_random[label]) >= n_train:
                    break

        else:
            for n, (cos, c) in enumerate(sorted(concepts_by_distance)):
                if (len(available_concepts) < step_size) or (step_size < 1):
                    self.train_random[label].append(c)
                    syns = self.synonyms_by_word[c]
                    for s in syns:
                        self.train_random[label].append(s)
                if n % step_size == 0:
                    self.train_random[label].append(c)
                    syns = self.synonyms_by_word[c]
                    for s in syns:
                        self.train_random[label].append(s)
                if len(self.train_random[label]) >= n_train:
                    break

    def extend_with_plurals(self, set_name):

        test_extended = defaultdict(list)
        train_extended = defaultdict(list)

        labels = ['pos', 'neg']
        if set_name == 'random':
            train = self.train_random
            test = self.test_random
        else:
            train = self.train
            test = self.test

        for l in labels:
            print(l)
            for c in train[l]:
                pl = self.sg_pl_mapping[c]
                train_extended[l].extend([c, pl])
                # update hyp dict
                self.concept_hyp_dict[pl] = self.concept_hyp_dict[c]

            for c in test[l]:
                pl = self.sg_pl_mapping[c]
                test_extended[l].extend([c, pl])
                # update hyp dict
                self.concept_hyp_dict[pl] = self.concept_hyp_dict[c]

        return train_extended, test_extended

    def get_cosine_dict_extended(self):

        all_concepts = []
        for c in self.concepts:
            all_concepts.append(c)
            all_concepts.append(self.sg_pl_mapping[c])

        all_concepts_pos = []
        for c in self.label_dict['pos']:
            all_concepts_pos.append(c)
            all_concepts_pos.append(self.sg_pl_mapping[c])

        concept_cosine_dict = dict()
        centroid, oov = utils.get_centroid(all_concepts_pos, self.model)
        distance_concept_list, oov = utils.get_distances_to_centroid(centroid, all_concepts, self.model)
        for cosine, wf, concept in distance_concept_list:
            concept_cosine_dict[concept] = cosine
        return concept_cosine_dict

    def data_to_file(self, model_name, split_name):

        prop = self.prop
        dir_path = f'../data/train_test_splits/{split_name}/{model_name}/{prop}/'
        os.makedirs(dir_path, exist_ok=True)

        # record the seeds
        if split_name.startswith('random-words-seed'):
            seed_path = f'../data/train_test_splits/{split_name}/{model_name}/{prop}/seed.text'
            with open(seed_path, 'w') as outfile:
                outfile.write('random seed word for positive examples: ' + self.seed)

        if split_name.startswith('random-words-seed'):
            split_dict = dict([('test', self.test_random_seed), ('train', self.train_random_seed)])
        elif split_name == 'random-words':
            split_dict = dict([('test', self.test_random), ('train', self.train_random)])
        elif split_name.startswith('random-words-no-dist'):
            split_dict = dict([('test', self.test_random_no_dist), ('train', self.train_random_no_dist)])
        else:
            split_dict = dict([('test', self.test), ('train', self.train)])

        split_label_dict = defaultdict(list)
        for data_split, example_dict in split_dict.items():
            for label, concepts in example_dict.items():
                for concept in concepts:
                    d = dict()
                    d['word'] = concept
                    d['label'] = label
                    if concept in self.concept_hyp_dict:
                        hyp_dict = self.concept_hyp_dict[concept]
                    else:
                        print('no hyp for', concept, data_split, label, split_name)
                    d.update(hyp_dict)
                    if concept in self.concept_cosine_dict:
                        d['cosine_centroid'] = self.concept_cosine_dict[concept]
                    else:
                        d['cosine_centroid'] = 'oov'
                    split_label_dict[data_split].append(d)

        for data_split, data in split_label_dict.items():
            path_split = f'{dir_path}{data_split}-syn_control_{self.synonymy_control}-pl_extension_{self.pl_extension}.csv'
            fieldnames = data[0].keys()
            with open(path_split, 'w') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                for d in data:
                    if 'hyp_rel' not in d.keys():
                        print(d)
                    writer.writerow(d)
