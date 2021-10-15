
# model_name='googlenews'
# model_path='../../../../Data/dsm/word2vec/GoogleNews-vectors-negative300.bin'
# model_dist_name='giga_corpus-google-wiki_corpus'

model_name='wiki_corpus'
model_path='/Users/piasommerauer/Data/DSM/corpus_exploration/wiki_full/trained_for_analysis_June2021/sgns_pinit1/sgns_rand_pinit1.words'
model_dist_name='giga_corpus-google-wiki_corpus'

# model_name='giga_corpus'
# model_path='/Users/piasommerauer/Data/DSM/corpus_exploration/giga_full/sgns_pinit1/sgns_rand_pinit1.words'
# model_dist_name='giga_corpus-google-wiki_corpus'

echo test
echo 'Runinng classification  for' $model_name $model_path $model_dist_name

python classify.py $model_name $model_path $model_dist_name
python compare_results.py $model_name $model_dist_name
python aggregate_mlp_results.py $model_name $model_dist_name
echo 'Calculating similarities'
python get_similarities.py $model_name $model_path $model_dist_name
echo 'Finished experiments'
