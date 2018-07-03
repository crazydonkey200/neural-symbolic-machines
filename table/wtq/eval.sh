DATA_DIR=$HOME"/projects/data/wikitable/"
INPUT_DIR=$DATA_DIR"processed_input/preprocess_14/"
SPLIT_DIR=$INPUT_DIR"data_split_1/"
python ../experiment.py \
       --eval_only \
       --experiment_to_eval=$1 \
       --output_dir=$DATA_DIR"output" \
       --experiment_name="eval_"$1 \
       --eval_file=$INPUT_DIR"test_split.jsonl" \
       --show_log \
       --eval_beam_size=1 \
       --alsologtostdout
