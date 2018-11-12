# Introduction

## Neural Symbolic Machines (NSM)

Neural Symbolic Machines is a framework to integrate neural networks and symbolic representations using reinforcement learning. 

<div align="middle"><img src="https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/images/nsm.png" width="80%" ></div>


## Applications

The framework can be used to learn semantic parsing and program synthesis from weak supervision (e.g., question-answer pairs), which is easier to collect and more flexible than full supervision (e.g., question-program pairs). Applications include virtual assistant, natural language interface to database, human-robot interaction, etc. It has been used to <a href="https://arxiv.org/abs/1611.00020">learn semantic parsers on Freebase<a> and <a href="https://arxiv.org/abs/1807.02322">natural language interfaces to database tables<a>. 

<div align="middle"><img src="https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/images/ap.png" width="80%"></div>


## Memory Augmented Policy Optimization (MAPO)

We use <a href="https://arxiv.org/abs/1807.02322">Memory Augmented Policy Optimization (MAPO)</a> to train NSM. It is a new policy optimization method that uses a memory buffer of promising trajectories to accelerate and stabilize policy gradient training. It is well suited for deterministic environments with discrete actions, for example, structured prediction and combinatorial optimization. 

<div align="middle"><img src="https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/images/mapo.png" width="80%"></div>

## Distributed Actor-Learner Architecture

Our implementation uses a distributed actor-learner architecture that utilizes multiple CPUs and GPUs for scalable training, similar to the one introduced in <a href="https://arxiv.org/abs/1802.01561">the IMPALA paper from DeepMind</a>. 

<div align="middle"><img src="https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/images/al.png" width="80%"></div>


# Dependencies

- Python 2.7
- TensorFlow>=1.7
- Other required packages are summarized in `requirements.txt`.

# Quick start

## Setup AWS instance
Start a g3.8xlarge instance with “Deep Learning AMI (Ubuntu) Version 10.0” image. (The experiments are conducted using this type of instance and image, you will need to adjust the configurations in scripts to run on other instances.)

Open port (for example, 6000-6010) in the security group for tensorboard. Instructions:
https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-tensorboard.html

ssh into the instance.

## Download the data and install the dependencies 
```
mkdir ~/projects
cd ~/projects/
git clone https://github.com/crazydonkey200/neural-symbolic-machines.git 

cd ~/projects/neural-symbolic-machines/
./aws_setup.sh
```

## Running experiments and monitor with tensorboard

### Start WikiTable experiment
```
screen -S wtq
source activate tensorflow_p27
cd ~/projects/neural-symbolic-machines/table/wtq/
./run.sh mapo your_experiment_name
```
This script trains the model for 30k steps and evaluates the checkpoint with the highest dev accuracy on the test set. It takes about 2.5 hrs to finish.

All the data about this experiment will be saved in `~/projects/data/wikitable/output/your_experiment_name`, and the evaluation result would be saved in `~/projects/data/wikitable/output/eval_your_experiment_name`.

You could also evaluate a trained model on the dev set or test set using
```
./eval.sh your_experiment_name dev

./eval.sh your_experiment_name test
```

### Start tensorboard to monitor WikiTable experiment
```
screen -S tb
source activate tensorflow_p27
cd  ~/projects/data/wikitable/
tensorboard --logdir=output
```
To see the tensorboard, in the browser, go to 
[your AWS public DNS]:6006
`avg_return_1` is the main metric (accuracy). 


### Start WikiSQL experiment.  
```
screen -S ws
source activate tensorflow_p27
cd ~/projects/neural-symbolic-machines/table/wikisql/
./run.sh mapo your_experiment_name
```
This script trains the model for 15k steps and evaluates the checkpoint with the highest dev accuracy on the test set. It takes about 6.5 hrs to finish. 

All the data about this experiment will be saved in `~/projects/data/wikisql/output/your_experiment_name`, and the evaluation result would be saved in `~/projects/data/wikisql/output/eval_your_experiment_name`.

You could also evaluate a trained model on the dev set or test set using
```
./eval.sh your_experiment_name dev

./eval.sh your_experiment_name test
```

### Start tensorboard to monitor WikiSQL experiment
```
screen -S tb
source activate tensorflow_p27
cd  ~/projects/data/wikisql/
tensorboard --logdir=output
```
To see the tensorboard, in the browser, go to 
[your AWS public DNS]:6006
`avg_return_1` is the main metric (accuracy). 

## Example outputs

Example learning curves for WikiTable (left) and WikiSQL (right) experiments (0.9 smoothing):
<img src="https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/images/wikitable_curve.png" width="50%"><img src="https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/images/wikisql_curve.png" width="50%">


## Citation
If you use the code in your research, please cite:

    @misc{1807.02322,
    Author = {Chen Liang and Mohammad Norouzi and Jonathan Berant and Quoc Le and Ni Lao},
    Title = {Memory Augmented Policy Optimization for Program Synthesis and Semantic Parsing},
    Year = {2018},
    Eprint = {arXiv:1807.02322},
    }

    @inproceedings{liang2017neural,
      title={Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision},
      author={Liang, Chen and Berant, Jonathan and Le, Quoc and Forbus, Kenneth D and Lao, Ni},
      booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      volume={1},
      pages={23--33},
      year={2017}
    }

