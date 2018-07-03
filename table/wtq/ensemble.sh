EVAL_OUTPUT=$1
ENSEMBLE_OUPUT=/tmp/ensemble_predictions.txt
python ensemble.py --eval_output_dir=$EVAL_OUTPUT --ensemble_output=$ENSEMBLE_OUPUT
python evaluator.py --t $HOME/projects/data/wikitable/raw_input/WikiTableQuestions/tagged/data/ $ENSEMBLE_OUPUT
