ROOT=$HOME"/projects/"
DATA_DIR=$ROOT"data/wikitable/"
python wtq_preprocess.py \
       --raw_input_dir=$DATA_DIR"raw_input" \
       --processed_input_dir=$DATA_DIR"processed_input/wtq_preprocess" \
       --max_n_tokens_for_num_prop=10 \
       --min_frac_for_ordered_prop=0.2 \
       --use_prop_match_count_feature \
       --expand_entities \
       --process_conjunction \
       --anonymize_datetime_and_number_entities \
       --alsologtostderr
