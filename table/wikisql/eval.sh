DATA_DIR=$HOME"/projects/data/wikisql/"
INPUT_DIR=$DATA_DIR"processed_input/preprocess_2/"
SPLIT_DIR=$INPUT_DIR
python ../experiment.py \
       --eval_only \
       --experiment_name="eval_"$1 \
       --experiment_to_eval=$1 \
       --output_dir=$DATA_DIR"output" \
       --eval_file=$INPUT_DIR"test_split.jsonl" \
       --executor="wikisql" \
       --alsologtostdout
