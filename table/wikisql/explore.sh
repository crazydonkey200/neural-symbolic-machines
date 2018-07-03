ROOT=$HOME"/projects/"
DATA_DIR=$ROOT"data/wikisql/"
python ../random_explore.py \
       --train_file_tmpl=$DATA_DIR"processed_input/preprocess_2/train_split_shard_30-{}.jsonl" \
       --table_file=$DATA_DIR"processed_input/preprocess_2/tables.jsonl" \
       --output_dir=$DATA_DIR"output" \
       --experiment_name="wikisql_random_exploration" \
       --n_explore_samples=50 \
       --max_n_exp=4 \
       --save_every_n=2 \
       --n_epoch=200 \
       --id_start=0 \
       --id_end=30 \
       --executor='wikisql' \
       --alsologtostderr
