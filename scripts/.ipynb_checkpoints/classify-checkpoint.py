from gensim.models import KeyedVectors
import sys

import utils_classify


def main():
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
    model_dict = model_dict

    model_name = model_dict['name']
    print(model_name)
    model = model_dict['model']

    pl_extension = False
    syn_control = False

    # running: wiki_full

    clf_names = ['lr', 'mlp']

    for clf_name in clf_names:
        print('##############')
        print(clf_name)
        print('##############')
        split_name = 'standard-cosine'
        print(split_name)
        props = utils_classify.get_properties(model_dist_name, split_name)
        df_dict = utils_classify.classify_set(model_name, model_dist_name, model, split_name,
                                              clf_name, props, pl_extension, syn_control)
        #
        #
        for n in range(10):
            split_name = f'random-words-seed-{n}'
            print(split_name)
            props = utils_classify.get_properties(model_dist_name, split_name)
            df_dict = utils_classify.classify_set(model_name, model_dist_name, model,
                                                   split_name, clf_name, props, pl_extension, syn_control)

        for n in range(10):
            split_name = f'random-words-no-dist-{n}'
            print(split_name)
            props = utils_classify.get_properties(model_dist_name, split_name)
            df_dict = utils_classify.classify_set(model_name, model_dist_name, model,
                                                  split_name, clf_name, props, pl_extension, syn_control)

        split_name = 'standard-cosine'
        print(split_name)
        props = utils_classify.get_properties(model_dist_name, split_name)
        df_dict = utils_classify.classify_random_vectors(model_name, model_dist_name, model, split_name,
                                                                   clf_name, props, pl_extension, syn_control)


if __name__ == '__main__':
    main()


