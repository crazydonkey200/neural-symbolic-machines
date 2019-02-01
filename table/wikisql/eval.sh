NAME=$1
CONFIG=$2
DATA_DIR=$HOME"/projects/data/wikisql/"
INPUT_DIR=$DATA_DIR"processed_input/preprocess_2/"
SPLIT_DIR=$INPUT_DIR
case $CONFIG in
    dev)
        echo "Evaluate on dev set."
        EVAL_FILE=$SPLIT_DIR"dev_split.jsonl"
        ;;
    test)
        echo "Evaluate on test set!"
        EVAL_FILE=$INPUT_DIR"test_split.jsonl"
        ;;
    *)
        echo "Usage: $0 experiment_name (dev|test)"
        exit 1
        ;;
esac
python ../experiment.py \
       --eval_only \
       --output_dir=$DATA_DIR"output" \
       --experiment_to_eval=$NAME \
       --experiment_name="eval_"$NAME \
       --eval_file=$EVAL_FILE \
       --executor="wikisql"
