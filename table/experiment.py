import shutil
import json
import time
import os
import sys
import re
import random
import cPickle as pkl
import functools
import pprint
import codecs
import multiprocessing

import copy
import numpy as np
import tensorflow as tf

import nsm
from nsm import data_utils
from nsm import env_factory
from nsm import graph_factory
from nsm import model_factory
from nsm import agent_factory
from nsm import executor_factory
from nsm import computer_factory
from nsm import word_embeddings

import utils


# FLAGS
FLAGS = tf.app.flags.FLAGS  


# Experiment name
tf.flags.DEFINE_string('output_dir', '', 'output folder.')
tf.flags.DEFINE_string('experiment_name', 'experiment',
                       'All outputs of this experiment is'
                       ' saved under a folder with the same name.')

# Random seed.
tf.app.flags.DEFINE_integer(
  'random_seed', -1,
  'Random seed for the NumPy and TF. Not used if setting to -1.')


# Tensorboard logging
tf.app.flags.DEFINE_string(
  'tb_log_dir', 'tb_log', 'Path for saving tensorboard logs.')


# Tensorflow model checkpoint.
tf.app.flags.DEFINE_string(
  'saved_model_dir', 'saved_model', 'Path for saving models.')
tf.app.flags.DEFINE_string(
  'best_model_dir', 'best_model', 'Path for saving best models.')
tf.app.flags.DEFINE_string(
  'init_model_path', '', 'Path for saving best models.')
tf.app.flags.DEFINE_string(
  'meta_graph_path', '', 'Path for meta graph.')
tf.app.flags.DEFINE_string('experiment_to_eval', '', '.')


# Model
## Computer
tf.app.flags.DEFINE_integer(
  'max_n_mem', 100, 'Max number of memory slots in the "computer".')
tf.app.flags.DEFINE_integer(
  'max_n_exp', 3, 'Max number of expressions allowed in a program.')
tf.app.flags.DEFINE_integer(
  'max_n_valid_indices', 100, 'Max number of valid tokens during decoding.')
tf.app.flags.DEFINE_bool(
  'use_cache', False, 'Use cache to avoid generating the same samples.')
tf.app.flags.DEFINE_string(
  'en_vocab_file', '', '.')
tf.app.flags.DEFINE_string(
  'executor', 'wtq', 'Which executor to use, wtq or wikisql.')


## neural network
tf.app.flags.DEFINE_integer(
  'hidden_size', 100, 'Number of hidden units.')
tf.app.flags.DEFINE_integer(
  'attn_size', 100, 'Size of attention vector.')
tf.app.flags.DEFINE_integer(
  'attn_vec_size', 100, 'Size of the vector parameter for computing attention.')
tf.app.flags.DEFINE_integer(
  'n_layers', 1, 'Number of layers in decoder.')
tf.app.flags.DEFINE_integer(
  'en_n_layers', 1, 'Number of layers in encoder.')
tf.app.flags.DEFINE_integer(
  'en_embedding_size', 100, 'Size of encoder input embedding.')
tf.app.flags.DEFINE_integer(
  'value_embedding_size', 300, 'Size of value embedding for the constants.')
tf.app.flags.DEFINE_bool(
  'en_bidirectional', False, 'Whether to use bidirectional RNN in encoder.')
tf.app.flags.DEFINE_bool(
  'en_attn_on_constants', False, '.')
tf.app.flags.DEFINE_bool(
  'use_pretrained_embeddings', False, 'Whether to use pretrained embeddings.')
tf.app.flags.DEFINE_integer(
  'pretrained_embedding_size', 300, 'Size of pretrained embedding.')


# Features
tf.app.flags.DEFINE_integer(
  'n_de_output_features', 1,
  'Number of features in decoder output softmax.')
tf.app.flags.DEFINE_integer(
  'n_en_input_features', 1,
  'Number of features in encoder inputs.')


# Data
tf.app.flags.DEFINE_string(
  'table_file', '', 'Path to the file of wikitables, a jsonl file.')
tf.app.flags.DEFINE_string(
  'train_file', '', 'Path to the file of training examples, a jsonl file.')
tf.app.flags.DEFINE_string(
  'dev_file', '', 'Path to the file of training examples, a jsonl file.')
tf.app.flags.DEFINE_string(
  'eval_file', '', 'Path to the file of test examples, a jsonl file.')
tf.app.flags.DEFINE_string(
  'embedding_file', '', 'Path to the file of pretrained embeddings, a npy file.')
tf.app.flags.DEFINE_string(
  'vocab_file', '', 'Path to the vocab file for the pretrained embeddings, a json file.')
tf.app.flags.DEFINE_string(
  'train_shard_dir', '', 'Folder containing the sharded training data.')
tf.app.flags.DEFINE_string(
  'train_shard_prefix', '', 'The prefix for the sharded files.')
tf.app.flags.DEFINE_integer(
  'n_train_shard', 90, 'Number of shards in total.')
tf.app.flags.DEFINE_integer(
  'shard_start', 0,
  'Start id of the shard to use.')
tf.app.flags.DEFINE_integer(
  'shard_end', 90, 'End id of the shard to use.')


# Load saved samples.
tf.app.flags.DEFINE_bool(
  'load_saved_programs', False,
  'Whether to use load saved programs from exploration.')
tf.app.flags.DEFINE_string(
  'saved_program_file', '', 'Saved program file.')


# Training
tf.app.flags.DEFINE_integer(
  'n_steps', 100000, 'Maximum number of steps in training.')
tf.app.flags.DEFINE_integer(
  'n_explore_samples', 1, 'Number of exploration samples per env per epoch.')
tf.app.flags.DEFINE_integer(
  'n_extra_explore_for_hard', 0, 'Number of exploration samples for hard envs.')
tf.app.flags.DEFINE_float(
  'learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
  'max_grad_norm', 5.0, 'Maximum gradient norm.')
tf.app.flags.DEFINE_float(
  'l2_coeff', 0.0, 'l2 regularization coefficient.')
tf.app.flags.DEFINE_float(
  'dropout', 0.0, 'Dropout rate.')
tf.app.flags.DEFINE_integer(
  'batch_size', 10, 'Model batch size.')
tf.app.flags.DEFINE_integer(
  'n_actors', 3, 'Number of actors for generating samples.')
tf.app.flags.DEFINE_integer(
  'save_every_n', -1,
  'Save model to a ckpt every n train steps, -1 means save every epoch.')
tf.app.flags.DEFINE_bool(
  'save_replay_buffer_at_end', True,
  'Whether to save the full replay buffer for each actor at the '
  'end of training or not')
tf.app.flags.DEFINE_integer(
  'log_samples_every_n_epoch', 0,
  'Log samples every n epochs.')
tf.app.flags.DEFINE_bool(
  'greedy_exploration', False,
  'Whether to use a greedy policy when doing systematic exploration.')
tf.app.flags.DEFINE_bool(
  'use_baseline', False,
  'Whether to use baseline during policy gradient.')
tf.app.flags.DEFINE_float(
  'min_prob', 0.0,
  ('Minimum probability of a negative'
   'example for it to be punished to avoid numerical issue.'))
tf.app.flags.DEFINE_float(
  'lm_loss_coeff', 0.0,
  'Weight for lm loss.')
tf.app.flags.DEFINE_float(
  'entropy_reg_coeff', 0.0,
  'Weight for entropy regularization.')
tf.app.flags.DEFINE_string(
  'optimizer', 'adam', '.')
tf.app.flags.DEFINE_float(
  'adam_beta1', 0.9, 'adam beta1 parameter.')
tf.app.flags.DEFINE_bool(
  'sample_other', False, 'Whether to use a greedy policy during training.')
tf.app.flags.DEFINE_bool(
  'use_replay_samples_in_train', False,
  'Whether to use replay samples for training.')
tf.app.flags.DEFINE_bool(
  'random_replay_samples', False,
  'randomly pick a replay samples as ML baseline.')
tf.app.flags.DEFINE_bool(
  'use_policy_samples_in_train', False,
  'Whether to use on-policy samples for training.')
tf.app.flags.DEFINE_bool(
  'use_nonreplay_samples_in_train', False,
  'Whether to use a negative samples for training.')
tf.app.flags.DEFINE_integer(
  'n_replay_samples', 5, 'Number of replay samples drawn.')
tf.app.flags.DEFINE_integer(
  'n_policy_samples', 5, 'Number of on-policy samples drawn.')
tf.app.flags.DEFINE_bool(
  'use_top_k_replay_samples', False,
  ('Whether to use the top k most probable (model probability) replay samples'
   ' or to sample from the replay samples.'))
tf.app.flags.DEFINE_bool(
  'use_top_k_policy_samples', False,
  ('Whether to use the top k most probable (from beam search) samples'
   ' or to sample from the replay samples.'))
tf.app.flags.DEFINE_float(
  'fixed_replay_weight', 0.5, 'Weight for replay samples between 0.0 and 1.0.')
tf.app.flags.DEFINE_bool(
  'use_replay_prob_as_weight', False,
  'Whether or not use replay probability as weight for replay samples.')
tf.app.flags.DEFINE_float(
  'min_replay_weight', 0.1, 'minimum replay weight.')
tf.app.flags.DEFINE_bool(
  'use_importance_sampling', False, '')
tf.app.flags.DEFINE_float(
  'ppo_epsilon', 0.2, '')
tf.app.flags.DEFINE_integer(
  'truncate_replay_buffer_at_n', 0,
  'Whether truncate the replay buffer to the top n highest prob trajs.')
tf.app.flags.DEFINE_bool(
  'use_trainer_prob', False,
  'Whether to supply all the replay buffer for training.')
tf.app.flags.DEFINE_bool(
  'show_log', False,
  'Whether to show logging info.')


# Eval
tf.app.flags.DEFINE_integer(
  'eval_beam_size', 5,
  'Beam size when evaluating on development set.')

tf.app.flags.DEFINE_integer(
  'eval_batch_size', 50,
  'Batch size when evaluating on development set.')

tf.app.flags.DEFINE_bool(
  'eval_only', False, 'only run evaluator.')

tf.app.flags.DEFINE_bool(
  'debug', False, 'Whether to output debug information.')


# Device placement.
tf.app.flags.DEFINE_bool(
  'train_use_gpu', False, 'Whether to output debug information.')
tf.app.flags.DEFINE_integer(
  'train_gpu_id', 0, 'Id of the gpu used for training.')

tf.app.flags.DEFINE_bool(
  'eval_use_gpu', False, 'Whether to output debug information.')
tf.app.flags.DEFINE_integer(
  'eval_gpu_id', 1, 'Id of the gpu used for eval.')

tf.app.flags.DEFINE_bool(
  'actor_use_gpu', False, 'Whether to output debug information.')
tf.app.flags.DEFINE_integer(
  'actor_gpu_start_id', 0,
  'Id of the gpu for the first actor, gpu for other actors will follow.')


# Testing
tf.app.flags.DEFINE_bool(
  'unittest', False, '.')

tf.app.flags.DEFINE_integer(
  'n_opt_step', 1, 'Number of optimization steps per training batch.')



def get_experiment_dir():
  experiment_dir = os.path.join(FLAGS.output_dir, FLAGS.experiment_name)
  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MkDir(FLAGS.output_dir)
  if not tf.gfile.IsDirectory(experiment_dir):
    tf.gfile.MkDir(experiment_dir)
  return experiment_dir  


def get_init_model_path():
  if FLAGS.init_model_path:
    return FLAGS.init_model_path
  elif FLAGS.experiment_to_eval:
    with open(os.path.join(
        FLAGS.output_dir,
        FLAGS.experiment_to_eval,
        'best_model_info.json'), 'r') as f:
      best_model_info = json.load(f)
      best_model_path = os.path.expanduser(
        best_model_info['best_model_path'])
      return best_model_path
  else:
    return ''


def get_saved_graph_config():
  if FLAGS.experiment_to_eval:
    with open(os.path.join(
        FLAGS.output_dir,
        FLAGS.experiment_to_eval,
        'graph_config.json'), 'r') as f:
      graph_config = json.load(f)
      return graph_config
  else:
    return None


def get_saved_experiment_config():
  if FLAGS.experiment_to_eval:
    with open(os.path.join(
        FLAGS.output_dir,
        FLAGS.experiment_to_eval,
        'experiment_config.json'), 'r') as f:
      experiment_config = json.load(f)
      return experiment_config
  else:
    return None
    

def show_samples(samples, de_vocab, env_dict=None):
  string = ''
  for sample in samples:
    traj = sample.traj
    actions = traj.actions
    obs = traj.obs
    pred_answer = traj.answer
    string += u'\n'
    env_name = traj.env_name
    string += u'env {}\n'.format(env_name)
    if env_dict is not None:
      string += u'question: {}\n'.format(env_dict[env_name].question_annotation['question'])
      string += u'answer: {}\n'.format(env_dict[env_name].question_annotation['answer'])
    tokens = []
    program = []
    for t, (a, ob) in enumerate(zip(actions, obs)):
      ob = ob[0]
      valid_tokens = de_vocab.lookup(ob.valid_indices, reverse=True)
      token = valid_tokens[a]
      program.append(token)
    program_str = ' '.join(program)
    if env_dict:
      program_str = unpack_program(program_str, env_dict[env_name])
    string += u'program: {}\n'.format(program_str)
    string += u'prediction: {}\n'.format(pred_answer)
    string += u'return: {}\n'.format(sum(traj.rewards))
    string += u'prob is {}\n'.format(sample.prob)
  return string


def collect_traj_for_program(env, program):
  env = env.clone()
  env.use_cache = False
  ob = env.start_ob

  for tk in program:
    valid_actions = list(ob[0].valid_indices)
    mapped_action = env.de_vocab.lookup(tk)
    try:
      action = valid_actions.index(mapped_action)
    except Exception:
      return None
    ob, _, _, _ = env.step(action)
  traj = agent_factory.Traj(
    obs=env.obs, actions=env.actions, rewards=env.rewards,
    context=env.get_context(), env_name=env.name, answer=env.interpreter.result)
  return traj


def unpack_program(program_str, env):
  ns = env.interpreter.namespace
  processed_program = []
  for tk in program_str.split():
    if tk[:1] == 'v' and tk in ns:
      processed_program.append(unicode(ns[tk]['value']))
    else:
      processed_program.append(tk)
  return ' '.join(processed_program)


def load_programs(envs, replay_buffer, fn):
  if not tf.gfile.Exists(fn):
    return 
  with open(fn, 'r') as f:
    program_dict = json.load(f)
  trajs = []
  n = 0
  total_env = 0
  n_found = 0
  for env in envs:
    total_env += 1
    found = False
    if env.name in program_dict:
      program_str_list = program_dict[env.name]
      n += len(program_str_list)
      env.cache._set = set(program_str_list)
      for program_str in program_str_list:
        program = program_str.split()
        traj = collect_traj_for_program(env, program)
        if traj is not None:
          trajs.append(traj)
          if not found:
            found = True
            n_found += 1
  tf.logging.info('@' * 100)
  tf.logging.info('loading programs from file {}'.format(fn))
  tf.logging.info('at least 1 solution found fraction: {}'.format(
    float(n_found) / total_env))
  replay_buffer.save_trajs(trajs)
  n_trajs_buffer = 0
  for k, v in replay_buffer._buffer.iteritems():
    n_trajs_buffer += len(v)
  tf.logging.info('{} programs in the file'.format(n))
  tf.logging.info('{} programs extracted'.format(len(trajs)))
  tf.logging.info('{} programs in the buffer'.format(n_trajs_buffer))
  tf.logging.info('@' * 100)


def get_program_shard_path(i):
  return os.path.join(
    FLAGS.saved_programs_dir, FLAGS.program_shard_prefix + str(i) + '.json')


def get_train_shard_path(i):
  return os.path.join(
    FLAGS.train_shard_dir, FLAGS.train_shard_prefix + str(i) + '.jsonl')


def load_jsonl(fn):
  result = []
  with open(fn, 'r') as f:
    for line in f:
      data = json.loads(line)
      result.append(data)
  return result


def create_envs(table_dict, data_set, en_vocab, embedding_model):
  all_envs = []
  t1 = time.time()
  if FLAGS.executor == 'wtq':
    score_fn = utils.wtq_score
    process_answer_fn = lambda x: x
    executor_fn = executor_factory.WikiTableExecutor
  elif FLAGS.executor == 'wikisql':
    score_fn = utils.wikisql_score
    process_answer_fn = utils.wikisql_process_answer
    executor_fn = executor_factory.WikiSQLExecutor
  else:
    raise ValueError('Unknown executor {}'.format(FLAGS.executor))

  for i, example in enumerate(data_set):
      if i % 100 == 0:
          tf.logging.info('creating environment #{}'.format(i))
      kg_info = table_dict[example['context']]
      executor = executor_fn(kg_info)
      api = executor.get_api()
      type_hierarchy = api['type_hierarchy']
      func_dict = api['func_dict']
      constant_dict = api['constant_dict']
      interpreter = computer_factory.LispInterpreter(
        type_hierarchy=type_hierarchy, 
        max_mem=FLAGS.max_n_mem, max_n_exp=FLAGS.max_n_exp, assisted=True)
      for v in func_dict.values():
          interpreter.add_function(**v)

      interpreter.add_constant(
        value=kg_info['row_ents'], type='entity_list', name='all_rows')

      de_vocab = interpreter.get_vocab()

      constant_value_embedding_fn = lambda x: utils.get_embedding_for_constant(
        x, embedding_model, embedding_size=FLAGS.pretrained_embedding_size)
      env = env_factory.QAProgrammingEnv(
        en_vocab, de_vocab, question_annotation=example,
        answer=process_answer_fn(example['answer']),
        constants=constant_dict.values(),
        interpreter=interpreter,
        constant_value_embedding_fn=constant_value_embedding_fn,
        score_fn=score_fn,
        name=example['id'])
      all_envs.append(env)
  return all_envs


def create_agent(graph_config, init_model_path,
                 pretrained_embeddings=None):
  tf.logging.info('Start creating and initializing graph')
  t1 = time.time()
  graph = graph_factory.MemorySeq2seqGraph(graph_config)
  graph.launch(init_model_path=init_model_path)
  t2 = time.time()
  tf.logging.info('{} sec used to create and initialize graph'.format(t2 - t1))

  tf.logging.info('Start creating model and agent')
  t1 = time.time()
  model = model_factory.MemorySeq2seqModel(graph, batch_size=FLAGS.batch_size)

  if pretrained_embeddings is not None:
    model.init_pretrained_embeddings(pretrained_embeddings)
  agent = agent_factory.PGAgent(model)
  t2 = time.time()
  tf.logging.info('{} sec used to create model and agent'.format(t2 - t1))
  return agent  
  

def init_experiment(fns, use_gpu=False, gpu_id='0'):
  dataset = []
  for fn in fns:
    dataset += load_jsonl(fn)
  tf.logging.info('{} examples in dataset.'.format(len(dataset)))
  tables = load_jsonl(FLAGS.table_file)
  table_dict = dict([(table['name'], table) for table in tables])
  tf.logging.info('{} tables.'.format(len(table_dict)))

  # Load pretrained embeddings.
  embedding_model = word_embeddings.EmbeddingModel(
    FLAGS.vocab_file, FLAGS.embedding_file)

  with open(FLAGS.en_vocab_file, 'r') as f:
    vocab = json.load(f)
  en_vocab = data_utils.Vocab([])
  en_vocab.load_vocab(vocab)
  tf.logging.info('{} unique tokens in encoder vocab'.format(
    len(en_vocab.vocab)))
  tf.logging.info('{} examples in the dataset'.format(len(dataset)))
  
  # Create environments.   
  envs = create_envs(table_dict, dataset, en_vocab, embedding_model)
  if FLAGS.unittest:
    envs = envs[:25]
  tf.logging.info('{} environments in total'.format(len(envs)))

  graph_config = get_saved_graph_config()
  if graph_config:
    # If evaluating an saved model, just load its graph
    # config.
    agent = create_agent(graph_config, get_init_model_path())
  else:
    if FLAGS.use_pretrained_embeddings:
      tf.logging.info('Using pretrained embeddings!')
      pretrained_embeddings = []
      for i in xrange(len(en_vocab.special_tks), en_vocab.size):
        pretrained_embeddings.append(
          utils.average_token_embedding(
            utils.find_tk_in_model(
              en_vocab.lookup(i, reverse=True), embedding_model),
            embedding_model,
            embedding_size=FLAGS.pretrained_embedding_size))
      pretrained_embeddings = np.vstack(pretrained_embeddings)
    else:
      pretrained_embeddings = None

    # Model configuration and initialization.
    de_vocab = envs[0].de_vocab
    n_mem = FLAGS.max_n_mem
    n_builtin = de_vocab.size - n_mem
    en_pretrained_vocab_size = en_vocab.size - len(en_vocab.special_tks)

    graph_config = {}
    graph_config['core_config'] = dict(
      max_n_valid_indices=FLAGS.max_n_valid_indices,
      n_mem=n_mem,
      n_builtin=n_builtin,
      use_attn=True, 
      attn_size=FLAGS.attn_size,
      attn_vec_size=FLAGS.attn_vec_size,
      input_vocab_size=de_vocab.size,
      en_input_vocab_size=en_vocab.size,
      hidden_size=FLAGS.hidden_size, n_layers=FLAGS.n_layers,
      en_hidden_size=FLAGS.hidden_size, en_n_layers=FLAGS.en_n_layers,
      en_use_embeddings=True,
      en_embedding_size=FLAGS.en_embedding_size,
      value_embedding_size=FLAGS.value_embedding_size,
      en_pretrained_vocab_size=en_pretrained_vocab_size,
      en_pretrained_embedding_size=FLAGS.pretrained_embedding_size,
      add_lm_loss=FLAGS.lm_loss_coeff > 0.0,
      en_bidirectional=FLAGS.en_bidirectional,
      en_attn_on_constants=FLAGS.en_attn_on_constants)

    graph_config['use_gpu'] = use_gpu
    graph_config['gpu_id'] = gpu_id

    graph_config['output_type'] = 'softmax'
    graph_config['output_config'] = dict(
      output_vocab_size=de_vocab.size, use_logits=True)
    aux_loss_list = [('ent_reg', FLAGS.entropy_reg_coeff),]

    if FLAGS.lm_loss_coeff > 0.0:
      aux_loss_list.append(('en_lm_loss', FLAGS.lm_loss_coeff))
    graph_config['train_config'] = dict(
      aux_loss_list=aux_loss_list,
      learning_rate=FLAGS.learning_rate,
      max_grad_norm=FLAGS.max_grad_norm,
      adam_beta1=FLAGS.adam_beta1,
      l2_coeff=FLAGS.l2_coeff,
      optimizer=FLAGS.optimizer, avg_loss_by_n=False)

    agent = create_agent(
      graph_config, get_init_model_path(),
      pretrained_embeddings=pretrained_embeddings)

  with open(os.path.join(get_experiment_dir(), 'graph_config.json'), 'w') as f:
    json.dump(graph_config, f, sort_keys=True, indent=2)
    
  return agent, envs


def compress_home_path(path):
  home_folder = os.path.expanduser('~')
  n = len(home_folder)
  if path[:n] == home_folder:
    return '~' + path[n:]
  else:
    return path


def create_experiment_config():
  experiment_config = get_saved_experiment_config()
  if experiment_config:
    FLAGS.embedding_file = os.path.expanduser(experiment_config['embedding_file'])
    FLAGS.vocab_file = os.path.expanduser(experiment_config['vocab_file'])
    FLAGS.en_vocab_file = os.path.expanduser(experiment_config['en_vocab_file'])
    FLAGS.table_file = os.path.expanduser(experiment_config['table_file'])

  experiment_config = {
    'embedding_file': compress_home_path(FLAGS.embedding_file),
    'vocab_file': compress_home_path(FLAGS.vocab_file),
    'en_vocab_file': compress_home_path(FLAGS.en_vocab_file),
    'table_file': compress_home_path(FLAGS.table_file)}

  return experiment_config


def run_experiment():
  print('=' * 100)
  if FLAGS.show_log:
    tf.logging.set_verbosity(tf.logging.INFO)

  experiment_dir = get_experiment_dir()
  if tf.gfile.Exists(experiment_dir):
    tf.gfile.DeleteRecursively(experiment_dir)
  tf.gfile.MkDir(experiment_dir)

  experiment_config = create_experiment_config()
  
  with open(os.path.join(
      get_experiment_dir(), 'experiment_config.json'), 'w') as f:
    json.dump(experiment_config, f)

  ckpt_queue = multiprocessing.Queue()
  train_queue = multiprocessing.Queue()
  eval_queue = multiprocessing.Queue()
  replay_queue = multiprocessing.Queue()

  run_type = 'evaluation' if FLAGS.eval_only else 'experiment'
  print('Start {} {}.'.format(run_type, FLAGS.experiment_name))
  print('The data of this {} is saved in {}.'.format(run_type, experiment_dir))

  if FLAGS.eval_only:
    print('Start evaluating the best model {}.'.format(get_init_model_path()))
  else:
    print('Start distributed training.')

  print('Start evaluator.')
  evaluator = Evaluator(
    'Evaluator',
    [FLAGS.eval_file if FLAGS.eval_only else FLAGS.dev_file])
  evaluator.start()

  if not FLAGS.eval_only:
    actors = []
    actor_shard_dict = dict([(i, []) for i in range(FLAGS.n_actors)])
    for i in xrange(FLAGS.shard_start, FLAGS.shard_end):
      actor_num = i % FLAGS.n_actors
      actor_shard_dict[actor_num].append(i)

    for k in xrange(FLAGS.n_actors):
      name = 'actor_{}'.format(k)
      actor = Actor(
        name, k, actor_shard_dict[k],
        ckpt_queue, train_queue, eval_queue, replay_queue)
      actors.append(actor)
      actor.start()
    print('Start {} actors.'.format(len(actors)))
    print('Start learner.')
    learner = Learner(
      'Learner', [FLAGS.dev_file], ckpt_queue,
      train_queue, eval_queue, replay_queue)
    learner.start()
    print('Use tensorboard to monitor the training progress (see README).')
    for actor in actors:
      actor.join()
    print('All actors finished')
    # Send learner the signal that all the actors have finished.
    train_queue.put(None)
    eval_queue.put(None)
    replay_queue.put(None)
    learner.join()
    print('Learner finished')

  evaluator.join()
  print('Evaluator finished')
  print('=' * 100)


class Actor(multiprocessing.Process):
    
  def __init__(
      self, name, actor_id, shard_ids, ckpt_queue, train_queue, eval_queue, replay_queue):
    multiprocessing.Process.__init__(self)
    self.ckpt_queue = ckpt_queue
    self.eval_queue = eval_queue
    self.train_queue = train_queue
    self.replay_queue = replay_queue
    self.name = name
    self.shard_ids = shard_ids
    self.actor_id = actor_id
      
  def run(self):
    if FLAGS.random_seed != -1:
      np.random.seed(FLAGS.random_seed)
      tf.set_random_seed(FLAGS.random_seed)
    agent, envs = init_experiment(
      [get_train_shard_path(i) for i in self.shard_ids],
      use_gpu=FLAGS.actor_use_gpu,
      gpu_id=str(self.actor_id + FLAGS.actor_gpu_start_id))
    graph = agent.model.graph
    current_ckpt = get_init_model_path()

    env_dict = dict([(env.name, env) for env in envs])
    replay_buffer = agent_factory.AllGoodReplayBuffer(agent, envs[0].de_vocab)

    # Load saved programs to warm start the replay buffer. 
    if FLAGS.load_saved_programs:
      load_programs(
        envs, replay_buffer, FLAGS.saved_program_file)

    if FLAGS.save_replay_buffer_at_end:
      replay_buffer_copy = agent_factory.AllGoodReplayBuffer(de_vocab=envs[0].de_vocab)
      replay_buffer_copy.program_prob_dict = copy.deepcopy(replay_buffer.program_prob_dict)

    i = 0
    while True:
      # Create the logging files. 
      if FLAGS.log_samples_every_n_epoch > 0 and i % FLAGS.log_samples_every_n_epoch == 0:
        f_replay = codecs.open(os.path.join(
          get_experiment_dir(), 'replay_samples_{}_{}.txt'.format(self.name, i)),
                               'w', encoding='utf-8')
        f_policy = codecs.open(os.path.join(
          get_experiment_dir(), 'policy_samples_{}_{}.txt'.format(self.name, i)),
                               'w', encoding='utf-8')
        f_train = codecs.open(os.path.join(
          get_experiment_dir(), 'train_samples_{}_{}.txt'.format(self.name, i)),
                              'w', encoding='utf-8')

      n_train_samples = 0
      if FLAGS.use_replay_samples_in_train:
        n_train_samples += FLAGS.n_replay_samples

      if FLAGS.use_policy_samples_in_train and FLAGS.use_nonreplay_samples_in_train:
        raise ValueError(
          'Cannot use both on-policy samples and nonreplay samples for training!')
        
      if FLAGS.use_policy_samples_in_train or FLAGS.use_nonreplay_samples_in_train:
        # Note that nonreplay samples are drawn by rejection
        # sampling from on-policy samples.
        n_train_samples += FLAGS.n_policy_samples

      # Make sure that all the samples from the env batch
      # fits into one batch for training.
      if FLAGS.batch_size < n_train_samples:
        raise ValueError(
            'One batch have to at least contain samples from one environment.')

      env_batch_size = FLAGS.batch_size / n_train_samples
      
      env_iterator = data_utils.BatchIterator(
        dict(envs=envs), shuffle=True,
        batch_size=env_batch_size)

      for j, batch_dict in enumerate(env_iterator):
        batch_envs = batch_dict['envs']
        tf.logging.info('=' * 50)
        tf.logging.info('{} iteration {}, batch {}: {} envs'.format(
            self.name, i, j, len(batch_envs)))
  
        t1 = time.time()
        # Generate samples with cache and save to replay buffer.
        t3 = time.time()
        n_explore = 0
        for _ in xrange(FLAGS.n_explore_samples):
          explore_samples = agent.generate_samples(
            batch_envs, n_samples=1, use_cache=FLAGS.use_cache,
            greedy=FLAGS.greedy_exploration)
          replay_buffer.save(explore_samples)
          n_explore += len(explore_samples)

        if FLAGS.n_extra_explore_for_hard > 0:
          hard_envs = [env for env in batch_envs
                       if not replay_buffer.has_found_solution(env.name)]
          if hard_envs:
            for _ in xrange(FLAGS.n_extra_explore_for_hard):
              explore_samples = agent.generate_samples(
                hard_envs, n_samples=1, use_cache=FLAGS.use_cache,
                greedy=FLAGS.greedy_exploration)
              replay_buffer.save(explore_samples)
              n_explore += len(explore_samples)

        t4 = time.time()
        tf.logging.info('{} sec used generating {} exploration samples.'.format(
          t4 - t3, n_explore))

        tf.logging.info('{} samples saved in the replay buffer.'.format(
          replay_buffer.size))
        
        t3 = time.time()
        replay_samples = replay_buffer.replay(
          batch_envs, FLAGS.n_replay_samples,
          use_top_k=FLAGS.use_top_k_replay_samples,
          agent=None if FLAGS.random_replay_samples else agent,
          truncate_at_n=FLAGS.truncate_replay_buffer_at_n)
        t4 = time.time()
        tf.logging.info('{} sec used selecting {} replay samples.'.format(
          t4 - t3, len(replay_samples)))
          
        t3 = time.time()
        if FLAGS.use_top_k_policy_samples:
          if FLAGS.n_policy_samples == 1:
            policy_samples = agent.generate_samples(
              batch_envs, n_samples=FLAGS.n_policy_samples,
              greedy=True)
          else:
            policy_samples = agent.beam_search(
              batch_envs, beam_size=FLAGS.n_policy_samples)
        else:
          policy_samples = agent.generate_samples(
            batch_envs, n_samples=FLAGS.n_policy_samples,
            greedy=False)
        t4 = time.time()
        tf.logging.info('{} sec used generating {} on-policy samples'.format(
          t4-t3, len(policy_samples)))

        t2 = time.time()
        tf.logging.info(
          ('{} sec used generating replay and on-policy samples,'
           ' {} iteration {}, batch {}: {} envs').format(
            t2-t1, self.name, i, j, len(batch_envs)))

        t1 = time.time()
        self.eval_queue.put((policy_samples, len(batch_envs)))
        self.replay_queue.put((replay_samples, len(batch_envs)))

        assert (FLAGS.fixed_replay_weight >= 0.0 and FLAGS.fixed_replay_weight <= 1.0)

        if FLAGS.use_replay_prob_as_weight:
          new_samples = []
          for sample in replay_samples:
            name = sample.traj.env_name
            if name in replay_buffer.prob_sum_dict:
              replay_prob = max(
                replay_buffer.prob_sum_dict[name], FLAGS.min_replay_weight)
            else:
              replay_prob = 0.0
            scale = replay_prob
            new_samples.append(
              agent_factory.Sample(
                traj=sample.traj,
                prob=sample.prob * scale))
          replay_samples = new_samples
        else:
          replay_samples = agent_factory.scale_probs(
            replay_samples, FLAGS.fixed_replay_weight)

        replay_samples = sorted(
          replay_samples, key=lambda x: x.traj.env_name)

        policy_samples = sorted(
          policy_samples, key=lambda x: x.traj.env_name)

        if FLAGS.use_nonreplay_samples_in_train:
          nonreplay_samples = []
          for sample in policy_samples:
            if not replay_buffer.contain(sample.traj):
              nonreplay_samples.append(sample)

        replay_buffer.save(policy_samples)

        def weight_samples(samples):
          if FLAGS.use_replay_prob_as_weight:
            new_samples = []
            for sample in samples:
              name = sample.traj.env_name
              if name in replay_buffer.prob_sum_dict:
                replay_prob = max(
                  replay_buffer.prob_sum_dict[name],
                  FLAGS.min_replay_weight)
              else:
                replay_prob = 0.0
              scale = 1.0 - replay_prob
              new_samples.append(
                agent_factory.Sample(
                  traj=sample.traj,
                  prob=sample.prob * scale))
          else:
            new_samples = agent_factory.scale_probs(
              samples, 1 - FLAGS.fixed_replay_weight)
          return new_samples

        train_samples = []
        if FLAGS.use_replay_samples_in_train:
          if FLAGS.use_trainer_prob:
            replay_samples = [
              sample._replace(prob=None) for sample in replay_samples]
          train_samples += replay_samples

        if FLAGS.use_policy_samples_in_train:
          train_samples += weight_samples(policy_samples)

        if FLAGS.use_nonreplay_samples_in_train:
          train_samples += weight_samples(nonreplay_samples)
        
        train_samples = sorted(train_samples, key=lambda x: x.traj.env_name)
        tf.logging.info('{} train samples'.format(len(train_samples)))

        if FLAGS.use_importance_sampling:
          step_logprobs = agent.compute_step_logprobs(
            [s.traj for s in train_samples])
        else:
          step_logprobs = None

        if FLAGS.use_replay_prob_as_weight:
          n_clip = 0
          for env in batch_envs:
            name = env.name
            if (name in replay_buffer.prob_sum_dict and
                replay_buffer.prob_sum_dict[name] < FLAGS.min_replay_weight):
              n_clip += 1
          clip_frac = float(n_clip) / len(batch_envs)
        else:
          clip_frac = 0.0
  
        self.train_queue.put((train_samples, step_logprobs, clip_frac))
        t2 = time.time()
        tf.logging.info(
          ('{} sec used preparing and enqueuing samples, {}'
           ' iteration {}, batch {}: {} envs').format(
             t2-t1, self.name, i, j, len(batch_envs)))

        t1 = time.time()
        # Wait for a ckpt that still exist or it is the same
        # ckpt (no need to load anything).
        while True:
          new_ckpt = self.ckpt_queue.get()
          new_ckpt_file = new_ckpt + '.meta'
          if new_ckpt == current_ckpt or tf.gfile.Exists(new_ckpt_file):
            break
        t2 = time.time()
        tf.logging.info('{} sec waiting {} iteration {}, batch {}'.format(
          t2-t1, self.name, i, j))

        if new_ckpt != current_ckpt:
          # If the ckpt is not the same, then restore the new
          # ckpt.
          tf.logging.info('{} loading ckpt {}'.format(self.name, new_ckpt))
          t1 = time.time()          
          graph.restore(new_ckpt)
          t2 = time.time()
          tf.logging.info('{} sec used {} restoring ckpt {}'.format(
            t2-t1, self.name, new_ckpt))
          current_ckpt = new_ckpt

        if FLAGS.log_samples_every_n_epoch > 0 and i % FLAGS.log_samples_every_n_epoch == 0:
          f_replay.write(show_samples(replay_samples, envs[0].de_vocab, env_dict))
          f_policy.write(show_samples(policy_samples, envs[0].de_vocab, env_dict))
          f_train.write(show_samples(train_samples, envs[0].de_vocab, env_dict))

      if FLAGS.log_samples_every_n_epoch > 0 and i % FLAGS.log_samples_every_n_epoch == 0:
        f_replay.close()
        f_policy.close()
        f_train.close()

      if agent.model.get_global_step() >= FLAGS.n_steps:
	if FLAGS.save_replay_buffer_at_end:
	  all_replay = os.path.join(get_experiment_dir(),
				    'all_replay_samples_{}.txt'.format(self.name))
	with codecs.open(all_replay, 'w', encoding='utf-8') as f:
	 samples = replay_buffer.all_samples(envs, agent=None)
	 samples = [s for s in samples if not replay_buffer_copy.contain(s.traj)]
	 f.write(show_samples(samples, envs[0].de_vocab, None))

        tf.logging.info('{} finished'.format(self.name))
        return
      i += 1


def select_top(samples):
  top_dict = {}
  for sample in samples:
    name = sample.traj.env_name
    prob = sample.prob
    if name not in top_dict or prob > top_dict[name].prob:
      top_dict[name] = sample    
  return agent_factory.normalize_probs(top_dict.values())


def beam_search_eval(agent, envs, writer=None):
    env_batch_size = FLAGS.eval_batch_size
    env_iterator = data_utils.BatchIterator(
      dict(envs=envs), shuffle=False,
      batch_size=env_batch_size)
    dev_samples = []
    dev_samples_in_beam = []
    for j, batch_dict in enumerate(env_iterator):
      t1 = time.time()
      batch_envs = batch_dict['envs']
      tf.logging.info('=' * 50)
      tf.logging.info('eval, batch {}: {} envs'.format(j, len(batch_envs)))
      new_samples_in_beam = agent.beam_search(
        batch_envs, beam_size=FLAGS.eval_beam_size)
      dev_samples_in_beam += new_samples_in_beam
      tf.logging.info('{} samples in beam, batch {}.'.format(
        len(new_samples_in_beam), j))
      t2 = time.time()
      tf.logging.info('{} sec used in evaluator batch {}.'.format(t2 - t1, j))

    # Account for beam search where the beam doesn't
    # contain any examples without error, which will make
    # len(dev_samples) smaller than len(envs).
    dev_samples = select_top(dev_samples_in_beam)
    dev_avg_return, dev_avg_len = agent.evaluate(
      dev_samples, writer=writer, true_n=len(envs))
    tf.logging.info('{} samples in non-empty beam.'.format(len(dev_samples)))
    tf.logging.info('true n is {}'.format(len(envs)))
    tf.logging.info('{} questions in dev set.'.format(len(envs)))
    tf.logging.info('{} dev avg return.'.format(dev_avg_return))
    tf.logging.info('dev: avg return: {}, avg length: {}.'.format(
      dev_avg_return, dev_avg_len))

    return dev_avg_return, dev_samples, dev_samples_in_beam


class Evaluator(multiprocessing.Process):
    
  def __init__(self, name, fns):
    multiprocessing.Process.__init__(self)
    self.name = name
    self.fns = fns

  def run(self):
    if FLAGS.random_seed != -1:
      np.random.seed(FLAGS.random_seed)
      tf.set_random_seed(FLAGS.random_seed)
    agent, envs = init_experiment(self.fns, FLAGS.eval_use_gpu, gpu_id=str(FLAGS.eval_gpu_id))
    for env in envs:
      env.punish_extra_work = False
    graph = agent.model.graph
    dev_writer = tf.summary.FileWriter(os.path.join(
      get_experiment_dir(), FLAGS.tb_log_dir, 'dev'))
    best_dev_avg_return = 0.0
    best_model_path = ''
    best_model_dir = os.path.join(get_experiment_dir(), FLAGS.best_model_dir)
    if not tf.gfile.Exists(best_model_dir):
      tf.gfile.MkDir(best_model_dir)
    i = 0
    current_ckpt = get_init_model_path()
    env_dict = dict([(env.name, env) for env in envs])
    while True:
      t1 = time.time()
      tf.logging.info('dev: iteration {}, evaluating {}.'.format(i, current_ckpt))

      dev_avg_return, dev_samples, dev_samples_in_beam = beam_search_eval(
        agent, envs, writer=dev_writer)
      
      if dev_avg_return > best_dev_avg_return:
        best_model_path = graph.save(
          os.path.join(best_model_dir, 'model'),
          agent.model.get_global_step())
        best_dev_avg_return = dev_avg_return
        tf.logging.info('New best dev avg returns is {}'.format(best_dev_avg_return))
        tf.logging.info('New best model is saved in {}'.format(best_model_path))
        with open(os.path.join(get_experiment_dir(), 'best_model_info.json'), 'w') as f:
          result = {'best_model_path': compress_home_path(best_model_path)}
          if FLAGS.eval_only:
            result['best_eval_avg_return'] = best_dev_avg_return
          else:
            result['best_dev_avg_return'] = best_dev_avg_return
          json.dump(result, f)

      if FLAGS.eval_only:
        # Save the decoding results for further. 
        dev_programs_in_beam_dict = {}
        for sample in dev_samples_in_beam:
          name = sample.traj.env_name
          program = agent_factory.traj_to_program(sample.traj, envs[0].de_vocab)
          answer = sample.traj.answer
          if name in dev_programs_in_beam_dict:
            dev_programs_in_beam_dict[name].append((program, answer, sample.prob))
          else:
            dev_programs_in_beam_dict[name] = [(program, answer, sample.prob)]

        t3 = time.time()
        with open(
            os.path.join(get_experiment_dir(), 'dev_programs_in_beam_{}.json'.format(i)),
            'w') as f:
          json.dump(dev_programs_in_beam_dict, f)
        t4 = time.time()
        tf.logging.info('{} sec used dumping programs in beam in eval iteration {}.'.format(
          t4 - t3, i))

        t3 = time.time()
        with codecs.open(
            os.path.join(
              get_experiment_dir(), 'dev_samples_{}.txt'.format(i)),
            'w', encoding='utf-8') as f:
          for sample in dev_samples:
            f.write(show_samples([sample], envs[0].de_vocab, env_dict))
        t4 = time.time()
        tf.logging.info('{} sec used logging dev samples in eval iteration {}.'.format(
          t4 - t3, i))

      t2 = time.time()
      tf.logging.info('{} sec used in eval iteration {}.'.format(
        t2 - t1, i))

      if FLAGS.eval_only or agent.model.get_global_step() >= FLAGS.n_steps:
        tf.logging.info('{} finished'.format(self.name))
        if FLAGS.eval_only:
          print('Eval average return (accuracy) of the best model is {}'.format(
            best_dev_avg_return))
        else:
          print('Best dev average return (accuracy) is {}'.format(best_dev_avg_return))
          print('Best model is saved in {}'.format(best_model_path))
        return

      # Reload on the latest model.
      new_ckpt = None
      t1 = time.time()
      while new_ckpt is None or new_ckpt == current_ckpt:
        time.sleep(1)
        new_ckpt = tf.train.latest_checkpoint(
          os.path.join(get_experiment_dir(), FLAGS.saved_model_dir))
      t2 = time.time()
      tf.logging.info('{} sec used waiting for new checkpoint in evaluator.'.format(
        t2-t1))
      
      tf.logging.info('lastest ckpt to evaluate is {}.'.format(new_ckpt))
      tf.logging.info('{} loading ckpt {}'.format(self.name, new_ckpt))
      t1 = time.time()
      graph.restore(new_ckpt)
      t2 = time.time()
      tf.logging.info('{} sec used {} loading ckpt {}'.format(
        t2-t1, self.name, new_ckpt))
      current_ckpt = new_ckpt


class Learner(multiprocessing.Process):
    
  def __init__(
      self, name, fns, ckpt_queue,
      train_queue, eval_queue, replay_queue):
    multiprocessing.Process.__init__(self)
    self.ckpt_queue = ckpt_queue
    self.eval_queue = eval_queue
    self.train_queue = train_queue
    self.replay_queue = replay_queue
    self.name = name
    self.save_every_n = FLAGS.save_every_n
    self.fns = fns
      
  def run(self):
    if FLAGS.random_seed != -1:
      np.random.seed(FLAGS.random_seed)
      tf.set_random_seed(FLAGS.random_seed)
    # Writers to record training and replay information.
    train_writer = tf.summary.FileWriter(os.path.join(
      get_experiment_dir(), FLAGS.tb_log_dir, 'train'))
    replay_writer = tf.summary.FileWriter(os.path.join(
      get_experiment_dir(), FLAGS.tb_log_dir, 'replay'))
    saved_model_dir = os.path.join(get_experiment_dir(), FLAGS.saved_model_dir)
    if not tf.gfile.Exists(saved_model_dir):
      tf.gfile.MkDir(saved_model_dir)
    agent, envs = init_experiment(self.fns, FLAGS.train_use_gpu, gpu_id=str(FLAGS.train_gpu_id))
    agent.train_writer = train_writer
    graph = agent.model.graph
    current_ckpt = get_init_model_path()

    i = 0
    n_save = 0
    while True:
      tf.logging.info('Start train step {}'.format(i))
      t1 = time.time()
      train_samples, behaviour_logprobs, clip_frac  = self.train_queue.get()
      eval_samples, eval_true_n = self.eval_queue.get()
      replay_samples, replay_true_n = self.replay_queue.get()
      t2 = time.time()
      tf.logging.info('{} secs used waiting in train step {}.'.format(
        t2-t1, i))
      t1 = time.time()
      n_train_samples = 0
      if FLAGS.use_replay_samples_in_train:
        n_train_samples += FLAGS.n_replay_samples
      if FLAGS.use_policy_samples_in_train and FLAGS.use_nonreplay_samples_in_train:
        raise ValueError(
          'Cannot use both on-policy samples and nonreplay samples for training!')
      if FLAGS.use_policy_samples_in_train:
        n_train_samples += FLAGS.n_policy_samples

      if train_samples:
        if FLAGS.use_trainer_prob:
          train_samples = agent.update_replay_prob(
            train_samples, min_replay_weight=FLAGS.min_replay_weight)
        for _ in xrange(FLAGS.n_opt_step):
          agent.train(
            train_samples,
            parameters=dict(en_rnn_dropout=FLAGS.dropout,rnn_dropout=FLAGS.dropout),
            use_baseline=FLAGS.use_baseline,
            min_prob=FLAGS.min_prob,
            scale=n_train_samples,
            behaviour_logprobs=behaviour_logprobs,
            use_importance_sampling=FLAGS.use_importance_sampling,
            ppo_epsilon=FLAGS.ppo_epsilon,
            de_vocab=envs[0].de_vocab,
            debug=FLAGS.debug)

      avg_return, avg_len = agent.evaluate(
        eval_samples, writer=train_writer, true_n=eval_true_n,
        clip_frac=clip_frac)
      tf.logging.info('train: avg return: {}, avg length: {}.'.format(
        avg_return, avg_len))
      avg_return, avg_len = agent.evaluate(
        replay_samples, writer=replay_writer, true_n=replay_true_n)
      tf.logging.info('replay: avg return: {}, avg length: {}.'.format(avg_return, avg_len))
      t2 = time.time()
      tf.logging.info('{} sec used in training train iteration {}, {} samples.'.format(
        t2-t1, i, len(train_samples)))
      i += 1
      if i % self.save_every_n == 0:
        t1 = time.time()
        current_ckpt = graph.save(
          os.path.join(saved_model_dir, 'model'),
          agent.model.get_global_step())
        t2 = time.time()
        tf.logging.info('{} sec used saving model to {}, train iteration {}.'.format(
          t2-t1, current_ckpt, i))
        self.ckpt_queue.put(current_ckpt)
        if agent.model.get_global_step() >= FLAGS.n_steps:
          t1 = time.time()
          while True:
            train_data = self.train_queue.get()
            _ = self.eval_queue.get()
            _ = self.replay_queue.get()
            self.ckpt_queue.put(current_ckpt)
            # Get the signal that all the actors have
            # finished.
            if train_data is None:
              t2 = time.time()
              tf.logging.info('{} finished, {} sec used waiting for actors'.format(
                self.name, t2-t1))
              return
      else:
        # After training on one set of samples, put one ckpt
        # back so that the ckpt queue is always full.
        self.ckpt_queue.put(current_ckpt)


def main(unused_argv):
  run_experiment()


if __name__ == '__main__':  
    tf.app.run()
