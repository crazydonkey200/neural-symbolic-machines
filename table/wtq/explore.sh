ROOT=$HOME"/projects/"
DATA_DIR=$ROOT"data/wikitable/"
python mp_random_explore.py \
       --train_file_tmpl=$DATA_DIR"processed_input/preprocess_14/data_split_1/train_split_shard_90-{}.jsonl" \
       --table_file=$DATA_DIR"processed_input/preprocess_14/tables.jsonl" \
       --output_dir=$DATA_DIR"output" \
       --experiment_name="random_explore_0" \
       --n_explore_samples=50 \
       --save_every_n=5 \
       --n_epoch=200 \
       --id_start=0 \
       --id_end=1 \
       --alsologtostderr
