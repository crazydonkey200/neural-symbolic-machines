"A collections of environments of sequence generations tasks."

from __future__ import division
from __future__ import print_function

import collections
import pprint

import nlp_utils
import tf_utils
import computer_factory

import bloom_filter
import tensorflow as tf


class Environment(object):
  """Environment with OpenAI Gym like interface."""
  
  def step(self, action):
    """
    Args:
      action: an action to execute against the environment.

    Returns:
      observation:
      reward:
      done:
      info: 
    """
    raise NotImplementedError


# Use last action and the new variable's memory location as input.
ProgrammingObservation = collections.namedtuple(
  'ProgramObservation', ['last_actions', 'output', 'valid_actions'])


class QAProgrammingEnv(Environment):
  """An RL environment wrapper around an interpreter to
  learn to write programs based on question.
  """
  def __init__(self, en_vocab, de_vocab,
               question_annotation, answer,
               constant_value_embedding_fn, 
               score_fn, interpreter, constants=None,
               punish_extra_work=True,
               init_interp=True, trigger_words_dict=None,
               max_cache_size=1e4,
               name='qa_programming'):
    self.name=name
    self.en_vocab = en_vocab
    self.de_vocab = de_vocab
    self.end_action = self.de_vocab.end_id
    self.score_fn = score_fn
    self.interpreter = interpreter
    self.answer = answer
    self.question_annotation = question_annotation
    self.constant_value_embedding_fn = constant_value_embedding_fn
    self.constants = constants
    self.punish_extra_work = punish_extra_work
    self.error = False
    self.trigger_words_dict = trigger_words_dict
    tokens = question_annotation['tokens']

    en_inputs = en_vocab.lookup(tokens)
    self.n_builtin = len(de_vocab.vocab) - interpreter.max_mem
    self.n_mem = interpreter.max_mem
    self.n_exp = interpreter.max_n_exp
    max_n_constants = self.n_mem - self.n_exp

    constant_spans = []
    constant_values = []
    if constants is None:
      constants = []
    for c in constants:
      constant_spans.append([-1, -1])
      constant_values.append(c['value'])
      if init_interp:
        self.interpreter.add_constant(
          value=c['value'], type=c['type'])

    for entity in question_annotation['entities']:
      # Use encoder output at start and end (inclusive) step
      # to create span embedding.
      constant_spans.append(
        [entity['token_start'], entity['token_end'] - 1])
      constant_values.append(entity['value'])
      if init_interp:
        self.interpreter.add_constant(
          value=entity['value'], type=entity['type'])

    constant_value_embeddings = [
      constant_value_embedding_fn(value) for value in constant_values]

    if len(constant_values) > (self.n_mem - self.n_exp):
      tf.logging.info(
        'Not enough memory slots for example {}, which has {} constants.'.format(
          self.name, len(constant_values)))

    constant_spans = constant_spans[:max_n_constants]
    constant_value_embeddings = constant_value_embeddings[:max_n_constants]
    self.context = (en_inputs, constant_spans, constant_value_embeddings,
                    question_annotation['features'],
                    question_annotation['tokens'])

    # Create output features.
    prop_features = question_annotation['prop_features']
    self.id_feature_dict = {}
    for name, id in de_vocab.vocab.iteritems():
      self.id_feature_dict[id] = [0]
      if name in self.interpreter.namespace:
        val = self.interpreter.namespace[name]['value']
        if ((isinstance(val, str) or isinstance(val, unicode)) and
            val in prop_features):
          self.id_feature_dict[id] = prop_features[val]
      
    self.cache = SearchCache(name=name, max_elements=max_cache_size)
    self.use_cache = False
    self.reset()
    
  def get_context(self):
    return self.context

  def step(self, action, debug=False):
    self.actions.append(action)
    if debug:
      print('-' * 50)
      print(self.de_vocab.lookup(self.valid_actions, reverse=True))
      print('pick #{} valid action'.format(action))
      print('history:')
      print(self.de_vocab.lookup(self.mapped_actions, reverse=True))
      print('env: {}, cache size: {}'.format(self.name, len(self.cache._set)))
      print('obs')
      pprint.pprint(self.obs)

    if action < len(self.valid_actions) and action >= 0: 
      mapped_action = self.valid_actions[action]
    else:
      print('-' * 50)
      # print('env: {}, cache size: {}'.format(self.name, len(self.cache._set)))
      print('action out of range.')
      print('action:')
      print(action)
      print('valid actions:')
      print(self.de_vocab.lookup(self.valid_actions, reverse=True))
      print('pick #{} valid action'.format(action))
      print('history:')
      print(self.de_vocab.lookup(self.mapped_actions, reverse=True))
      print('obs')
      pprint.pprint(self.obs)
      print('-' * 50)
      mapped_action = self.valid_actions[action]
      
    self.mapped_actions.append(mapped_action)

    result = self.interpreter.read_token(
      self.de_vocab.lookup(mapped_action, reverse=True))

    self.done = self.interpreter.done
    # Only when the proram is finished and it doesn't have
    # extra work or we don't care, its result will be
    # scored, and the score will be used as reward. 
    if self.done and not (self.punish_extra_work and self.interpreter.has_extra_work()):
      reward = self.score_fn(self.interpreter.result, self.answer)
    else:
      reward = 0.0

    if self.done and self.interpreter.result == [computer_factory.ERROR_TK]:
      self.error = True

    if result is None or self.done:
      new_var_id = -1
    else:
      new_var_id = self.de_vocab.lookup(self.interpreter.namespace.last_var)
    valid_tokens = self.interpreter.valid_tokens()
    valid_actions = self.de_vocab.lookup(valid_tokens)

    # For each action, check the cache for the program, if
    # already tried, then not valid anymore.
    if self.use_cache:
      new_valid_actions = []
      cached_actions = []
      partial_program = self.de_vocab.lookup(self.mapped_actions, reverse=True)
      for ma in valid_actions:
        new_program = partial_program + [self.de_vocab.lookup(ma, reverse=True)]
        if not self.cache.check(new_program):
          new_valid_actions.append(ma)
        else:
          cached_actions.append(ma)
      valid_actions = new_valid_actions

    self.valid_actions = valid_actions
    self.rewards.append(reward)
    ob = (tf_utils.MemoryInputTuple(
      read_ind=mapped_action, write_ind=new_var_id, valid_indices=self.valid_actions),
          [self.id_feature_dict[a] for a in valid_actions])

    # If no valid actions are available, then stop.
    if not self.valid_actions:
      self.done = True
      self.error = True

    # If the program is not finished yet, collect the
    # observation.
    if not self.done:
      # Add the actions that are filtered by cache into the
      # training example because at test time, they will be
      # there (no cache is available).
      if self.use_cache:
        valid_actions = self.valid_actions + cached_actions
        true_ob = (tf_utils.MemoryInputTuple(
          read_ind=mapped_action, write_ind=new_var_id,
          valid_indices=valid_actions),
                   [self.id_feature_dict[a] for a in valid_actions])
        self.obs.append(true_ob)
      else:
        self.obs.append(ob)
    elif self.use_cache:
      # If already finished, save it in the cache.
      self.cache.save(self.de_vocab.lookup(self.mapped_actions, reverse=True))

    return ob, reward, self.done, {}
      #'valid_actions': valid_actions, 'new_var_id': new_var_id}

  def reset(self):
    self.actions = []
    self.mapped_actions = []
    self.rewards = []
    self.done = False
    valid_actions = self.de_vocab.lookup(self.interpreter.valid_tokens())
    if self.use_cache:
      new_valid_actions = []
      for ma in valid_actions:
        partial_program = self.de_vocab.lookup(
          self.mapped_actions + [ma], reverse=True)
        if not self.cache.check(partial_program):
          new_valid_actions.append(ma)
      valid_actions = new_valid_actions
    self.valid_actions = valid_actions
    self.start_ob = (tf_utils.MemoryInputTuple(
      self.de_vocab.decode_id, -1, valid_actions),
                     [self.id_feature_dict[a] for a in valid_actions])
    self.obs = [self.start_ob]

  def interactive(self):
    self.interpreter.interactive()
    print('reward is: %s' % score_fn(self.interpreter))

  def clone(self):
    new_interpreter = self.interpreter.clone()
    new = QAProgrammingEnv(
      self.en_vocab, self.de_vocab, score_fn=self.score_fn,
      question_annotation=self.question_annotation,
      constant_value_embedding_fn=self.constant_value_embedding_fn,
      constants=self.constants,
      answer=self.answer, interpreter=new_interpreter,
      init_interp=False)
    new.actions = self.actions[:]
    new.mapped_actions = self.mapped_actions[:]
    new.rewards = self.rewards[:]
    new.obs = self.obs[:]
    new.done = self.done
    new.name = self.name
    # Cache is shared among all copies of this environment.
    new.cache = self.cache
    new.use_cache = self.use_cache
    new.valid_actions = self.valid_actions
    new.error = self.error
    new.id_feature_dict = self.id_feature_dict
    new.punish_extra_work = self.punish_extra_work
    new.trigger_words_dict = self.trigger_words_dict
    return new

  def show(self):
    program = ' '.join(
      self.de_vocab.lookup([o.read_ind for o in self.obs], reverse=True))
    valid_tokens = ' '.join(self.de_vocab.lookup(self.valid_actions, reverse=True))
    return 'program: {}\nvalid tokens: {}'.format(program, valid_tokens)


class SearchCache(object):
  def __init__(self, name, size=None, max_elements=1e4, error_rate=1e-8):
    self.name = name
    self.max_elements = max_elements
    self.error_rate = error_rate
    self._set = bloom_filter.BloomFilter(
      max_elements=max_elements, error_rate=error_rate)
    
  def check(self, tokens):
    return ' '.join(tokens) in self._set

  def save(self, tokens):
    string = ' '.join(tokens)
    self._set.add(string)
    
  def is_full(self):
    return '(' in self._set

  def reset(self):
    self._set = bloom_filter.BloomFilter(
      max_elements=self.max_elements, error_rate=self.error_rate)
