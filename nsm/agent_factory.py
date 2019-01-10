"""Implementation of RL agents."""

import collections
import os
import logging
import time
import shutil
import pprint
import heapq

import tensorflow as tf
import numpy as np

import data_utils
import graph_factory
import env_factory


# To suppress division by zero in np.log, because the
# behaviour np.log(0.0) = -inf is correct in this context.
np.warnings.filterwarnings('ignore')


def scale_probs(samples, scale):
  "Weight each samples with the weight. Reflected on probs."
  scaled_probs = [scale * s.prob for s in samples]
  new_samples = []
  for s, p in zip(samples, scaled_probs):
    new_samples.append(Sample(traj=s.traj, prob=p))
  return new_samples


def scale_rewards(samples, scale):
  "Weight each samples with the weight. Reflected on rewards."
  scaled_rewards = [list(scale * np.array(s.rewards)) for s in samples]
  new_samples = []
  for s, p in zip(samples, scaled_rewards):
    new_samples.append(sample._replace(traj=traj._replace(rewards=scaled_rewards)))
  return new_samples
  

def normalize_probs(samples, smoothing=1e-8):
  "Normalize the probability of the samples (in each env) to sum to 1.0."
  sum_prob_dict = {}
  for s in samples:
    name = s.traj.env_name
    if name in sum_prob_dict:
      sum_prob_dict[name] += s.prob + smoothing
    else:
      sum_prob_dict[name] = s.prob + smoothing
  new_samples = []
  for s in samples:
    new_prob = (s.prob + smoothing) / sum_prob_dict[s.traj.env_name]
    new_samples.append(Sample(traj=s.traj, prob=new_prob))
  return new_samples


def compute_baselines(returns, probs, env_names):
  "Compute baseline for samples."
  baseline_dict = {}
  sum_dict = {}
  for ret, p, name in zip(returns, probs, env_names):
    if name not in sum_dict:
      sum_dict[name] = ret[-1] * p
    else:
      sum_dict[name] += ret[-1] * p
  for name, _ in sum_dict.iteritems():
    # Assume probabilities are already normalized.
    baseline_dict[name] = sum_dict[name]
  return baseline_dict


class PGAgent(object):
  "Agent trained by policy gradient."
  def __init__(self, model, train_writer=None, discount_factor=1.0):

    self.model = model
    self.discount_factor = discount_factor

    self.train_writer = train_writer

    self.monitor_graph = graph_factory.MonitorGraph()
    for name in ['avg_return', 'min_return', 'max_return', 'std_return',
                 'avg_len', 'min_len', 'max_len', 'std_len',
                 'clip_frac']:
      self.monitor_graph.add_scalar_monitor(name, dtype=tf.float32)

    self.monitor_graph.launch()

  def generate_samples(self, envs, n_samples=1, greedy=False,
                       use_cache=False, filter_error=True): #, parameters=None):
    """Returns Actions, rewards, obs, other info."""
    samples = sampling(
      self.model, envs, n_samples=n_samples, greedy=greedy, use_cache=use_cache,
      filter_error=filter_error)
    return samples

  def beam_search(self, envs=None, beam_size=1, use_cache=False, greedy=False):
    """Returns Actions, rewards, obs and probs."""    
    samples = beam_search(self.model, envs, beam_size=beam_size,
                          use_cache=use_cache, greedy=greedy)
    return samples

  def update_replay_prob(self, samples, min_replay_weight=0.0):
    """Update the probability of the replay samples and recompute the weights (prob)."""
    prob_sum_dict = {}
    trajs_to_update = [
      sample.traj for sample in samples if sample.prob is None]
    new_probs = self.compute_probs(trajs_to_update)
    for traj, prob in zip(trajs_to_update, new_probs):
      name = traj.env_name
      if name in prob_sum_dict:
        prob_sum_dict[name] += prob
      else:
        prob_sum_dict[name] = prob

    i = 0
    new_samples = []
    for sample in samples:
      name = sample.traj.env_name
      if name in prob_sum_dict:
        w = max(prob_sum_dict[name], min_replay_weight)
      else:
        w = 0.0
      if sample.prob is None:
        new_samples.append(
          sample._replace(prob=new_probs[i] * w / prob_sum_dict[name]))
        i += 1
      else:
        prob = sample.prob * (1.0 - w)
        new_samples.append(sample._replace(prob=prob))
    assert i == len(trajs_to_update)

    return new_samples

  def train(self, samples, use_baseline=False,
            debug=False, parameters=None, min_prob=0.0, scale=1.0,
            behaviour_logprobs=None, use_importance_sampling=False,
            ppo_epsilon=0.2, weight_by_target_prob=True,
            de_vocab=None):

    trajs = [s.traj for s in samples]
    probs = [s.prob for s in samples]
    env_names = [t.env_name for t in trajs]
    returns = [compute_returns(t.rewards, self.discount_factor) for t in trajs]

    if use_baseline:
      baseline_dict = compute_baselines(returns, probs, env_names)
      advantages = []
      for t, r in zip(trajs, returns):
        advantages.append(
          list(np.array(r) - baseline_dict[t.env_name]))
    else:
      advantages = returns

    obs = [t.obs for t in trajs]
    actions = [t.actions for t in trajs]
    rewards = [t.rewards for t in trajs]
    context = [t.context for t in trajs]

    weights = [list(np.array(ad) * p * scale) for ad, p in zip(advantages, probs)]

    if not use_importance_sampling and min_prob > 0.0:
      model_probs = self.compute_probs(trajs)
      for i, mp in enumerate(model_probs):
        # If the probabiity of the example is already small
        # and the advantage is still negative, then stop
        # punishing it because it might cause numerical
        # instability.
        if mp < min_prob and len(advantages[i]) > 0 and advantages[i][0] < 0.0:
          weights[i] = len(weights[i]) * [0.0]

    if use_importance_sampling:
      assert behaviour_logprobs is not None
      new_weights = []
      target_logprobs = self.compute_step_logprobs(trajs)
      ratio = [list(np.exp(np.array(tlp) - np.array(blp))) for blp, tlp in
               zip(behaviour_logprobs, target_logprobs)]
      if debug:
        print 'ratio: '
        pprint.pprint(ratio)
      # print 'advantages: '
      # print advantages      
      for i in xrange(len(weights)):
        new_ws = []
        for w, ad, blp, tlp in zip(
            weights[i], advantages[i],
            behaviour_logprobs[i], target_logprobs[i]):
          if weight_by_target_prob:
            if ad > 0.0:
              new_ws.append(max(np.exp(tlp), 0.1) * ad)
            else:
              new_ws.append(np.exp(tlp) * ad)
          elif ad >= 0 and (tlp - blp) > np.log(1 + ppo_epsilon):
            new_ws.append(0.0)
          elif ad < 0 and (tlp - blp) < np.log(1 - ppo_epsilon):
            new_ws.append(0.0)
          elif tlp > np.log(1 - 1e-4) and ad > 0.0:
            new_ws.append(0.0)
          elif tlp < np.log(1e-4) and ad < 0.0:
            new_ws.append(0.0)
          else:
           new_ws.append(w * np.exp(tlp - blp))
        if min_prob > 0.0 and sum(target_logprobs[i]) < np.log(min_prob):
          new_ws = [0.0] * len(new_ws)
        new_weights.append(new_ws)
      weights = new_weights
      if debug:
        print 'new_weights: '
        pprint.pprint(weights)

    if debug:
      print('+' * 50)      
      model_probs = self.compute_probs(trajs)
      print('scale: {}, min_prob: {}'.format(scale, min_prob))
      for i, (name, c, o, a, r, ad, p, mp, w, traj) in enumerate(zip(
          env_names, context, obs, actions, returns, advantages,
          probs, model_probs, weights, trajs)):
        print(('sample {}, name: {}, return: {}, advantage: {}, '
               'prob: {}, model prob: {}, weight: {}').format(
                 i, name, r[0], ad[0], p, mp, w[0]))
        if de_vocab is not None:
          print(' '.join(traj_to_program(traj, de_vocab)))
      print('+' * 50)
      
    self.model.train(obs, actions, weights=weights, context=context,
                     parameters=parameters, writer=self.train_writer)

  def compute_probs(self, trajs):
    obs = [s.obs for s in trajs]
    actions = [s.actions for s in trajs]
    context = [s.context for s in trajs]
    probs = self.model.compute_probs(obs, actions, context=context)
    return probs

  def compute_step_logprobs(self, trajs):
    obs = [s.obs for s in trajs]
    actions = [s.actions for s in trajs]
    context = [s.context for s in trajs]
    logprobs = self.model.compute_step_logprobs(obs, actions, context=context)
    return logprobs
    
  def evaluate(self, samples, n_samples=1, verbose=0, writer=None, true_n=None,
               clip_frac=0.0):
    "Evaluate the agent on the envs."

    trajs = [s.traj for s in samples]
    actions = [t.actions for t in trajs]
    probs = [s.prob for s in samples]
  
    returns = [compute_returns(t.rewards, self.discount_factor)[0] for t in trajs]

    avg_return, std_return, max_return, min_return, n_w = compute_weighted_stats(
      returns, probs)

    if true_n is not None:
      # Account for the fact that some environment doesn't
      # generate any valid samples, but we still need to
      # consider them when computing returns.
      new_avg_return = avg_return * n_w / true_n
      tf.logging.info(
        'avg return adjusted from {} to {} based on true n'.format(
          avg_return, new_avg_return))
      avg_return = new_avg_return
    
    lens = [len(acs) for acs in actions]
    avg_len, std_len, max_len, min_len, _ = compute_weighted_stats(lens, probs)
    
    if verbose > 0:
      print ('return: avg {} +- {}, max {}, min {}\n'
             'length: avg {} +- {}, max {}, min {}').format(
               avg_return, return_array.std(),
               return_array.max(), return_array.min(),
               avg_len, len_array.std(), len_array.max(), len_array.min())

    if writer is not None:
      feed_dict = dict(
        avg_return=avg_return, max_return=max_return,
        min_return=min_return, std_return=std_return,
        avg_len=avg_len, max_len=max_len,
        min_len=min_len, std_len=std_len,
        clip_frac=clip_frac)
      self.write_to_monitor(feed_dict, writer)
      # summary = self.monitor_graph.generate_summary(feed_dict)
      # writer.add_summary(summary, self.model.get_global_step())
      # writer.flush()
    return avg_return, avg_len

  def write_to_monitor(self, feed_dict, writer):
    summary = self.monitor_graph.generate_summary(feed_dict)
    writer.add_summary(summary, self.model.get_global_step())
    writer.flush()


def compute_weighted_stats(array, weight):
  n = len(array)
  if n < 1:
    return (0.0, 0.0, 0.0, 0.0, 0.0)
  sum_ = 0.0
  min_ = array[0]
  max_ = array[0]
  n_w = sum(weight)
  for a, w in zip(array, weight):
    min_ = min([min_, a])
    max_ = max([max_, a])
    sum_ += a * w
  mean = sum_ / n_w
  sum_square_std = 0.0
  for a in array:
    sum_square_std += (a - mean) ** 2 * w
  std = np.sqrt(sum_square_std / n_w)
  return (mean, std, max_, min_, n_w)


def compute_returns(rewards, discount_factor=1.0):
  "Compute returns of a trace (sum of discounted rewards)."
  returns = []
  t = len(rewards)
  returns = [0.0] * t
  sum_return_so_far = 0.0
  for i in xrange(t):
    returns[-i-1] = sum_return_so_far * discount_factor + rewards[-i-1]
    sum_return_so_far = returns[-1-i]
  return returns


def compute_td_errors(values, rewards, discount_factor=1.0, td_n=0):
  "Compute TD errors."
  td_errors = []
  td_n += 1
  backup_values = compute_backup_values(values, rewards, discount_factor, td_n)
  for vs, bvs in zip(values, backup_values):
    td_errors.append((np.array(bvs) - np.array(vs)).tolist())
  return td_errors


def compute_backup_values(values, rewards, discount_factor=1.0, n_steps=1):
  "Compute backup value."
  backup_values = []
  for vs, rs in zip(values, rewards):
    bvs = []
    T = len(vs)
    for i in xrange(T):
      end = min(i + n_steps, T)
      if end == T:
        bv = 0.0
      else:
        bv = vs[end] * (discount_factor ** (end - i))
      for t in xrange(i, end):
        bv += rs[t] * (discount_factor ** (t-i))
      bvs.append(bv)
    backup_values.append(bvs)
  return backup_values


class ReplayBuffer(object):

  def save(self, samples):
    pass

  def replay(self, envs):
    pass


class AllGoodReplayBuffer(ReplayBuffer):
  def __init__(self, agent=None, de_vocab=None, discount_factor=1.0, debug=False):
    self._buffer = dict()
    self.discount_factor = discount_factor
    self.agent = agent
    self.de_vocab = de_vocab
    self.program_prob_dict = dict()
    self.prob_sum_dict = dict()

  def has_found_solution(self, env_name):
    return env_name in self._buffer and self._buffer[env_name]

  def contain(self, traj):
    name = traj.env_name
    if name not in self.program_prob_dict:
      return False
    program = traj_to_program(traj, self.de_vocab)
    program_str = u' '.join(program)
    if program_str in self.program_prob_dict[name]:
      return True
    else:
      return False

  @property
  def size(self):
    n = 0
    for _, v in self._buffer.iteritems():
      n += len(v)
    return n
      
  def save(self, samples):
    trajs = [s.traj for s in samples]
    self.save_trajs(trajs)

  def save_trajs(self, trajs):
    total_returns = [
      compute_returns(t.rewards, self.discount_factor)[0] for t in trajs]
    for t, return_ in zip(trajs, total_returns):
      name = t.env_name
      program = traj_to_program(t, self.de_vocab)
      program_str = ' '.join(program)
      if (return_ > 0.5 and
          len(program) > 0 and
          (program[-1] == self.de_vocab.end_tk) and
          (not (name in self.program_prob_dict and
                (program_str in self.program_prob_dict[name])))):
        if name in self.program_prob_dict:
          self.program_prob_dict[name][program_str] = True
        else:
          self.program_prob_dict[name] = {program_str: True}
        if name in self._buffer:
          self._buffer[name].append(t)
        else:
          self._buffer[name] = [t]

  def all_samples(self, envs, agent=None):
    select_env_names = set([e.name for e in envs])
    trajs = []
    # Collect all the trajs for the selected envs.
    for name in select_env_names:
      if name in self._buffer:
        trajs += self._buffer[name]
    if agent is None:
      # All traj has the same probability, since it will be
      # normalized later, we just assign them all as 1.0.
      probs = [1.0] * len(trajs)
    else:
      # Otherwise use the agent to compute the prob for each
      # traj.
      probs = agent.compute_probs(trajs)
    samples = [Sample(traj=t, prob=p) for t, p in zip(trajs, probs)]
    return samples

  def replay(self, envs, n_samples=1, use_top_k=False, agent=None, truncate_at_n=0):
    select_env_names = set([e.name for e in envs])
    trajs = []
    # Collect all the trajs for the selected envs.
    for name in select_env_names:
      if name in self._buffer:
        trajs += self._buffer[name]
    if agent is None:
      # All traj has the same probability, since it will be
      # normalized later, we just assign them all as 1.0.
      probs = [1.0] * len(trajs)
    else:
      # Otherwise use the agent to compute the prob for each
      # traj.
      probs = agent.compute_probs(trajs)

    # Put the samples into an dictionary keyed by env names.
    samples = [Sample(traj=t, prob=p) for t, p in zip(trajs, probs)]
    env_sample_dict = dict(
      [(name, []) for name in select_env_names
       if name in self._buffer])
    for s in samples:
      name = s.traj.env_name
      env_sample_dict[name].append(s)
      
    replay_samples = []
    for name, samples in env_sample_dict.iteritems():
      n = len(samples)
      # Truncated the number of samples in the selected
      # samples and in the buffer.
      if truncate_at_n > 0 and n > truncate_at_n:
        # Randomize the samples before truncation in case
        # when no prob information is provided and the trajs
        # need to be truncated randomly.
        np.random.shuffle(samples)
        samples = heapq.nlargest(
          truncate_at_n, samples, key=lambda s: s.prob)
        self._buffer[name] = [sample.traj for sample in samples]

      # Compute the sum of prob of replays in the buffer.
      self.prob_sum_dict[name] = sum([sample.prob for sample in samples])

      if use_top_k:
        # Select the top k samples weighted by their probs.
        selected_samples = heapq.nlargest(
          n_samples, samples, key=lambda s: s.prob)
        replay_samples += normalize_probs(selected_samples)
      else:
        # Randomly samples according to their probs.
        samples = normalize_probs(samples)
        selected_samples = [samples[i] for i in np.random.choice(
          len(samples), n_samples, p=[sample.prob for sample in samples])]
        replay_samples += [
          Sample(traj=s.traj, prob=1.0 / n_samples) for s in selected_samples]

    return replay_samples


def traj_to_program(traj, de_vocab):
  program = []
  for a, ob in zip(traj.actions, traj.obs):
    ob = ob[0]
    token = de_vocab.lookup(ob.valid_indices[a], reverse=True)
    program.append(token)
  return program
  
    
Traj = collections.namedtuple(
  'Traj', 'obs actions rewards context env_name answer')

Sample = collections.namedtuple(
  'Sample', 'traj prob')


def sampling(model, envs, temperature=1.0, use_encode=True,
             greedy=False, n_samples=1, debug=False, use_cache=False,
             filter_error=True):

  if not envs:
    raise ValueError('No environment provided!')

  if use_cache:
    # if already explored everything, then don't explore this environment anymore.
    envs = [env for env in envs if not env.cache.is_full()]

  duplicated_envs = []
  for env in envs:
    for i in range(n_samples):
      duplicated_envs.append(env.clone())
      
  envs = duplicated_envs
  
  for env in envs:
    env.use_cache = use_cache
  
  if use_encode:
    env_context = [env.get_context() for env in envs]
    encoded_context, initial_state = model.encode(env_context)
  else:
    # env_context = [None for env in envs]
    encoded_context, initial_state = None, None

  obs = [[env.start_ob] for env in envs]
  state = initial_state
  
  while True:
    outputs, state = model.step(obs, state, context=encoded_context)
    
    if greedy:
      actions = model.predict(cell_outputs=outputs)
    else:
      actions = model.sampling(
        cell_outputs=outputs, temperature=temperature)

    if debug:
      print '*' * 50
      print 'actions: '
      pprint.pprint(actions)
      print 'action_prob: '
      action_prob = model.predict_prob(cell_outputs=outputs)
      pprint.pprint(action_prob)
      print '*' * 50

    # Get rid of the time dimension so that actions is just one list.
    actions = [a[0] for a in actions]
    action_probs = model.predict_prob(cell_outputs=outputs)
    action_probs = [ap[0] for ap in action_probs]

    obs = []
    for env, action, p in zip(envs, actions, action_probs):
      try:
        ob, _, _, info = env.step(action)
        obs.append([ob])
      except IndexError:
        print p
        raise IndexError
    step_pairs = [
      x for x in zip(obs, state, encoded_context, envs) if not x[-1].done]
    if len(step_pairs) > 0:
      obs, state, encoded_context, envs = zip(*step_pairs)
      obs = list(obs)
      state = list(state)
      envs = list(envs)
      encoded_context = list(encoded_context)
      assert len(obs) == len(state)
      assert len(obs) == len(encoded_context)
      assert len(obs) == len(envs)
    else:
      break

  obs, actions, rewards = zip(*[(env.obs, env.actions, env.rewards)
                                for env in duplicated_envs])
  env_names = [env.name for env in duplicated_envs]
  answers = [env.interpreter.result for env in duplicated_envs]
  
  samples = []
  for i, env in enumerate(duplicated_envs):
    if not (filter_error and env.error):
      samples.append(
        Sample(
          traj=Traj(obs=env.obs, actions=env.actions, rewards=env.rewards,
                    context=env_context[i], env_name=env_names[i],
                    answer=answers[i]),
          prob=1.0 / n_samples))
  return samples


Hyph = collections.namedtuple('Hyph', ['state', 'env', 'score'])
Candidate = collections.namedtuple('Candidate', ['state', 'env', 'score', 'action'])


def beam_search(model, envs, use_encode=True, beam_size=1, debug=False, renorm=True,
                use_cache=False, filter_error=True, greedy=False):
  if use_cache:
    # if already explored everything, then don't explore this environment anymore.
    envs = [env for env in envs if not env.cache.is_full()]
    
  if use_encode:
    env_context = [env.get_context() for env in envs]
    encoded_context, initial_state = model.encode(env_context)
    env_context_dict = dict(
      [(env.name, env.get_context()) for env in envs])
    context_dict = dict(
      [(env.name, c) for env, c in zip(envs, encoded_context)])
    beam = [Hyph(s, env.clone(), 0.0)
            for env, s in zip(envs, initial_state)]
    state = initial_state
    context = encoded_context
  else:
    beam = [Hyph(None, env.clone(), 0.0) for env in envs]
    state = None
    context = None
    env_context_dict = dict(
      [(env.name, None) for env in envs])

  for hyp in beam:
    hyp.env.use_cache = use_cache      
    
  finished_dict = dict([(env.name, []) for env in envs])
  obs = [[h.env.start_ob] for h in beam]

  while beam:
    if debug:
      print '@' * 50
      print 'beam is'
      for h in beam:
        print 'env {}'.format(h.env.name)
        print h.env.show()
        print h.score
        print
  
    # Run the model for one step to get probabilities for new actions.
    outputs, state = model.step(obs, state, context=context)

    probs = model.predict_prob(outputs)
    scores = (np.log(np.array(probs)) + np.array([[[h.score]] for h in beam]))

    # Collect candidates.
    candidate_dict = {}
    for hyph, st, score in zip(beam, state, scores):
      env_name = hyph.env.name
      if env_name not in candidate_dict:
        candidate_dict[env_name] = []
      for action, s in enumerate(score[0]):
        if not s == -np.inf:
          candidate_dict[env_name].append(Candidate(st, hyph.env, s, action))

    if debug:
      print '*' * 20
      print 'candidates are'
      for k, v in candidate_dict.iteritems():      
        print 'env {}'.format(k)
        for x in v:
          print x.env.show()
          print x.action
          print x.score
          print type(x)
          print isinstance(x, Candidate)
          print
        
    # Collect the new beam.
    new_beam = []
    obs = []
    for env_name, candidates in candidate_dict.iteritems():
      # Find the top k from the union of candidates and
      # finished hypotheses.
      all_candidates = candidates + finished_dict[env_name]
      topk = heapq.nlargest(
        beam_size, all_candidates, key=lambda x: x.score)

      # Step the environment and collect the hypotheses into
      # new beam (unfinished hypotheses) or finished_dict
      finished_dict[env_name] = []
      for c in topk:
        if isinstance(c, Hyph):
          finished_dict[env_name].append(c)
        else:
          env = c.env.clone()
          #print 'action', c.action
          ob, _, done, info = env.step(c.action)
          #pprint.pprint(env.de_vocab.lookup(info['valid_actions'], reverse=True))
          #pprint.pprint(env.de_vocab.lookup(info['new_var_id'], reverse=True))
          new_hyph = Hyph(c.state, env, c.score)
          if not done:
            obs.append([ob])
            new_beam.append(new_hyph)
          else:
            if not (filter_error and new_hyph.env.error):
              finished_dict[env_name].append(new_hyph)

    if debug:
      print '#' * 20
      print 'finished programs are'
      for k, v in finished_dict.iteritems():
        print 'env {}'.format(k)
        for x in v:
          print x.env.show()
          print x.score
          print type(x)
          print isinstance(x, Hyph)
          print
            
    beam = new_beam
    
    if use_encode:
      state = [h.state for h in beam]
      context = [context_dict[h.env.name] for h in beam]
    else:
      state = None
      context = None

  final = []
  env_names = [env.name for env in envs]
  for name in env_names:
    sorted_final = sorted(
      finished_dict[name],
      key=lambda h: h.score, reverse=True)
    if greedy:
      # Consider the time when sorted_final is empty (didn't
      # find any programs without error).
      if sorted_final:
        final += [sorted_final[0]]
    else:
      final += sorted_final

  # Collect the training examples.
  obs, actions, rewards, env_names, scores = zip(
    *[(h.env.obs, h.env.actions, h.env.rewards, h.env.name, h.score)
      for h in final])
  answers = [h.env.interpreter.result for h in final]

  samples = []
  for i, name in enumerate(env_names):
    samples.append(
      Sample(
        traj=Traj(obs=obs[i], actions=actions[i], rewards=rewards[i],
                  context=env_context_dict[name], env_name=name,
                  answer=answers[i]),
        prob=np.exp(scores[i])))

  if renorm:
    samples = normalize_probs(samples)
  return samples


# A random agent.

class RandomAgent(object):
  def __init__(self, discount_factor=1.0):
    self.discount_factor = discount_factor

  def generate_samples(self, envs, n_samples=1, use_cache=False):      
    if use_cache:
      # if already explored everything, then don't explore this environment anymore.
      envs = [env for env in envs if not env.cache.is_full()]

    for env in envs:
      env.use_cache = use_cache
    
    duplicated_envs = []
    for env in envs:
      for i in range(n_samples):
        duplicated_envs.append(env.clone())

    envs = duplicated_envs

    for env in envs:
      ob = env.start_ob
      while not env.done:
        valid_actions = ob[0].valid_indices
        action = np.random.randint(0, len(valid_actions))
        ob, _, _, _ = env.step(action)

    env_context = [env.get_context() for env in envs]
    env_names = [env.name for env in envs]
    samples = []
    for i, env in enumerate(envs):
      samples.append(
        Sample(
          traj=Traj(obs=env.obs, actions=env.actions, rewards=env.rewards,
                    context=env_context[i], env_name=env_names[i]),
          prob=1.0 / n_samples))
    return samples

  def evaluate(self, samples):
    trajs = [s.traj for s in samples]
    actions = [t.actions for t in trajs]

    probs = [s.prob for s in samples]
  
    returns = [compute_returns(t.rewards, self.discount_factor)[0] for t in trajs]

    avg_return, std_return, max_return, min_return = compute_weighted_stats(
      returns, probs)

    lens = [len(acs) for acs in actions]
    avg_len, std_len, max_len, min_len = compute_weighted_stats(lens, probs)
    return avg_return, avg_len
