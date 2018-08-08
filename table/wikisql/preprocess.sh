ROOT=$HOME"/projects/"
DATA_DIR=$ROOT"data/wikisql/"
python preprocess.py \
       --raw_input_dir=$DATA_DIR"raw_input" \
       --processed_input_dir=$DATA_DIR"processed_input/wikisql_preprocess" \
       --alsologtostderr
