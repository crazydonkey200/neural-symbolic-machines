CONFIG=$1
NAME=$2
USE_NONREPLAY=nouse_nonreplay_samples_in_train
RANDOM_REPLAY=norandom_replay_samples
USE_REPLAY_PROB=nouse_replay_prob_as_weight
FIXED_REPLAY_WEIGHT=1.0
TOPK_REPLAY=nouse_top_k_replay_samples
USE_TRAINER_PROB=nouse_trainer_prob
TRUNCATE_AT_N=5
N_STEPS=15000
OUTPUT=output
case $CONFIG in
    mapo)
        echo mapo
        N_REPLAY=5
        TOPK_REPLAY=use_top_k_replay_samples
        USE_TRAINER_PROB=use_trainer_prob
        USE_NONREPLAY=use_nonreplay_samples_in_train
        USE_REPLAY_PROB=use_replay_prob_as_weight
        BS=750
        ;;
    mml)
        echo mml
        USE_TRAINER_PROB=use_trainer_prob
        N_REPLAY=5
        BS=625
        ;;
    iml)
        echo iml
        RANDOM_REPLAY=random_replay_samples
        N_REPLAY=5
        TRUNCATE_AT_N=0
        BS=625
        ;;
    hard_em)
        echo hard_em
        TOPK_REPLAY=use_top_k_replay_samples
        N_REPLAY=1
        BS=125
        ;;
    *)
        echo "Usage: $0 (mapo|mapo_enum|mml|iml|hard_em) experiment_name"
        exit 1
        ;;
esac
DATA_DIR="$HOME/projects/data/wikisql/"
INPUT_DIR=$DATA_DIR"processed_input/preprocess_2/"
SPLIT_DIR=$INPUT_DIR
python ../experiment.py \
       --output_dir=$DATA_DIR$OUTPUT \
       --experiment_name=$NAME \
       --n_actors=30 \
       --dev_file=$SPLIT_DIR"dev_split.jsonl" \
       --train_shard_dir=$SPLIT_DIR \
       --train_shard_prefix="train_split_shard_30-" \
       --shard_start=0 \
       --shard_end=30 \
       --load_saved_programs \
       --saved_program_file=$INPUT_DIR"all_train_saved_programs-1k_5.json" \
       --embedding_file=$DATA_DIR"raw_input/wikisql_glove_embedding_mat.npy" \
       --vocab_file=$DATA_DIR"raw_input/wikisql_glove_vocab.json" \
       --table_file=$INPUT_DIR"tables.jsonl" \
       --en_vocab_file=$INPUT_DIR"en_vocab_min_count_5.json" \
       --save_every_n=10 \
       --n_explore_samples=1 \
       --n_extra_explore_for_hard=9 \
       --use_cache \
       --batch_size=$BS \
       --dropout=0.0 \
       --hidden_size=200 \
       --attn_size=200 \
       --attn_vec_size=200 \
       --en_embedding_size=200 \
       --en_bidirectional \
       --n_layers=2 \
       --en_n_layers=2 \
       --use_pretrained_embeddings \
       --pretrained_embedding_size=300 \
       --value_embedding_size=300 \
       --learning_rate=0.001 \
       --n_policy_samples=1 \
       --n_replay_samples=$N_REPLAY \
       --use_replay_samples_in_train \
       --$USE_NONREPLAY \
       --$USE_REPLAY_PROB \
       --$TOPK_REPLAY \
       --fixed_replay_weight=$FIXED_REPLAY_WEIGHT \
       --$RANDOM_REPLAY \
       --$USE_TRAINER_PROB \
       --min_replay_weight=0.1 \
       --truncate_replay_buffer_at_n=$TRUNCATE_AT_N \
       --train_use_gpu \
       --train_gpu_id=0 \
       --eval_use_gpu \
       --eval_gpu_id=1 \
       --max_n_mem=60 \
       --max_n_valid_indices=60 \
       --max_n_exp=4 \
       --eval_beam_size=5 \
       --executor="wikisql" \
       --n_steps=$N_STEPS \
       --show_log
python ../experiment.py \
       --eval_only \
       --eval_use_gpu \
       --eval_gpu_id=0 \
       --experiment_name="eval_"$NAME \
       --experiment_to_eval=$NAME \
       --output_dir=$DATA_DIR$OUTPUT \
       --eval_file=$INPUT_DIR"test_split.jsonl" \
       --executor="wikisql"
