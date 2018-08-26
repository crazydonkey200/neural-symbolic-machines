"Implements several tensorflow graphs and capsulate them as Graph."

import abc
import six
import collections
import os
import pprint

import numpy as np
import tensorflow as tf

import tf_utils
import data_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RNN_CELL_DICT = dict(
  rnn=tf.contrib.rnn.RNNCell,
  lstm=tf.contrib.rnn.BasicLSTMCell,
  layernorm_lstm=tf.contrib.rnn.LayerNormBasicLSTMCell,
  gru=tf.contrib.rnn.GRUCell)

OPTIMIZER_DICT = dict(
  sgd=tf.train.GradientDescentOptimizer,
  adam=tf.train.AdamOptimizer,
  adagrad=tf.train.AdagradOptimizer,
  rmsprop=tf.train.RMSPropOptimizer)

ACTIVATION_DICT = dict(
  relu=tf.nn.relu, sigmoid=tf.nn.sigmoid,
  tanh=tf.nn.tanh)

# Bind a variable length tensor with its sequence_length.
SeqTensor = collections.namedtuple('SeqTensor', ['tensor', 'sequence_length'])


def with_graph_variable_scope(func):
  def func_wrapper(*args, **kwargs):
    self = args[0]
    with self._graph.as_default():
      pid = os.getpid()
      container_name = 'worker{}'.format(pid)
      # print(container_name)
      with self._graph.container(container_name):
        with tf.variable_scope(self.vs):
          return func(*args, **kwargs)
  return func_wrapper

  
class Graph(object):
  """A TensorFlow graph with simpler interface to interact with it.

  The neural network architecture (basically all the
  tensorflow code) should live within this class. A new
  architecture (for example, Seq2seq) should implement a new
  subclass (Seq2seqGraph).
  """
  def __init__(self, name, meta_graph_fn=''):
    self.node_dict = {'summaries': []}
    self._graph = tf.Graph()
    self.vs_name = name
    self.use_gpu = False
    with tf.variable_scope(name) as vs:
      self.vs = vs

  @with_graph_variable_scope
  def launch(self, init_model_path=''):
    "Launch and initialize the graph."
    if self.use_gpu:
      n_gpu = 1
    else:
      n_gpu = 0
    session_config = tf.ConfigProto(
      device_count={'GPU' : n_gpu},
      allow_soft_placement=False,
      log_device_placement=False,
    )
    tf.logging.info('number of gpu used {}'.format(n_gpu))

    self.session = tf.Session(
      graph=self._graph, config=session_config)
    self.saver = tf.train.Saver(tf.global_variables())
    if init_model_path:
      self._graph.finalize()
      self.saver.restore(self.session, init_model_path)
    else:
      init = tf.global_variables_initializer()
      self._graph.finalize()
      self.session.run(init)
    return self.session

  def restore(self, model_path):
    self.saver.restore(self.session, model_path)

  def save(self, model_path, global_step):
    return self.saver.save(self.session, model_path, global_step)
    
  def run(self, fetch_list, feed_dict, writer=None):
    """Main interface to interact with the tensorflow graph.

    Args:
      fetch_list: a list of names (strings) indicating
        the name of result operations.
      feed_dict: a dictionary with the names of the nodes as keys
        and the corresponding values that are fed as values.
      writer: a tensorflow summary writer

    Returns:
      outputs: a dictionary with the names in the fetch_list as
        keys, and the outputs from the executing graph as values.      
    """
    fetch_dict = dict([(name, self.node_dict[name])
                       for name in fetch_list if name in self.node_dict])

    if writer is not None:
      fetch_dict['summaries'] = self.node_dict['summaries']
      fetch_dict['global_step'] = self.node_dict['global_step']
      
    outputs = self.session.run(
      fetch_dict, map_dict(self.node_dict, feed_dict))
    
    if writer is not None:
      writer.add_summary(
        outputs['summaries'], outputs['global_step'])
      writer.flush()

    return outputs

  @with_graph_variable_scope
  def add_train(self, aux_loss_list=None, optimizer='adam',
                learning_rate=0.01, max_grad_norm=5.0,
                decay_after_n_steps=1000, decay_every_n_steps=1000,
                lr_decay_factor=1.0, avg_loss_by_n=False, debug=False,
                l2_coeff=0.0,
                adam_beta1=0.9,
                name='Training'):
    "Construct part of the graph that controlls training (SGD optimization)."
    self.node_dict['max_batch_size'] = tf.placeholder(tf.int32, None)
    with tf.variable_scope(name):
      all_summaries = []
      batch_size = tf.cast(self.node_dict['max_batch_size'],
                           dtype=tf.float32)
      loss = self.node_dict['loss'] / batch_size
      
      all_summaries.append(
        tf.summary.scalar(self.vs_name + '/' + 'loss', loss))
      total_loss = loss

      if aux_loss_list is not None:
        for loss_name, w in aux_loss_list:
          if w > 0.0:
            aux_loss = self.node_dict[loss_name]
            total_loss += aux_loss * w / batch_size
            aux_loss_summary = tf.summary.scalar(
              self.vs_name + '/' + loss_name, aux_loss)
            all_summaries.append(aux_loss_summary)

      if debug:
        total_loss= tf.Print(
          total_loss, [self.node_dict['sequence_loss']], message='seq_loss:', summarize=10000)

        total_loss= tf.Print(
          total_loss, [self.node_dict['weights'].tensor], message='weights:', summarize=10000)

        total_loss= tf.Print(
          total_loss, [self.node_dict['targets'].tensor], message='targets:', summarize=10000)

        total_loss= tf.Print(
          total_loss, [self.node_dict['probs'].tensor], message='probs:', summarize=10000)

        total_loss= tf.Print(
          total_loss, [self.node_dict['step_loss'].tensor],
          message='step_loss:', summarize=10000)

        total_loss= tf.Print(
          total_loss, [self.node_dict['logits'].tensor], message='logits:', summarize=10000)
      
      total_loss_summary = tf.summary.scalar(
        self.vs_name + '/' + 'total_loss', total_loss)
      all_summaries.append(total_loss_summary)
      
      lr = tf.Variable(
        float(learning_rate), trainable=False)

      new_lr = tf.placeholder(dtype=tf.float32, shape=None, name='new_lr')
      update_lr = lr.assign(new_lr)

      global_step = tf.Variable(0, trainable=False)
      decay_step = tf.maximum(
        0, global_step - decay_after_n_steps)
      decay_exponent = (tf.cast(decay_step, tf.float32) /
                        tf.cast(decay_every_n_steps, tf.float32))
      decay = lr_decay_factor ** decay_exponent
      decayed_lr = lr * decay
      lr_summary = tf.summary.scalar(self.vs_name + '/' + 'learning_rate', decayed_lr)
      all_summaries.append(lr_summary)
      
      params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.vs_name)
      n_params = 0
      tf.logging.info('trainable parameters:')
      for tv in params:
        n_tv_params = np.product(tv.get_shape().as_list())
        n_params += n_tv_params
        tf.logging.info('{}: {}'.format(tv.name, n_tv_params))
        if 'weights' in tv.name or 'kernel' in tv.name:
          total_loss += tf.reduce_sum(tf.nn.l2_loss(tv)) * l2_coeff
      tf.logging.info('total number of trainable parameters {}'.format(n_params))

      if optimizer == 'adam':
        tf.logging.info('adam beta1: {}'.format(adam_beta1))
        opt = OPTIMIZER_DICT[optimizer](decayed_lr, beta1=adam_beta1)
      else:
        opt = OPTIMIZER_DICT[optimizer](decayed_lr)

      gradients = tf.gradients(total_loss, params)
      clipped_gradients, grad_norm = tf.clip_by_global_norm(
        gradients, max_grad_norm)
      update = opt.apply_gradients(
        zip(clipped_gradients, params), global_step=global_step)
            
      grad_norm_summary = tf.summary.scalar(
        self.vs_name + '/' + 'grad_norm', grad_norm)
      all_summaries.append(grad_norm_summary)

      if debug:
        _, clipped_grad_norm = tf.clip_by_global_norm(
          clipped_gradients, max_grad_norm)
        clipped_grad_norm_summary = tf.summary.scalar(
          self.vs_name + '/' + 'clipped_grad_norm', clipped_grad_norm)
        n_summary = tf.summary.scalar(
          self.vs_name + '/' + 'n', self.node_dict['n'])
        seq_loss_summary = tf.summary.histogram(
          self.vs_name + '/' + 'seq_loss', self.node_dict['sequence_loss'])
        step_loss_summary = tf.summary.histogram(
          self.vs_name + '/' + 'step_loss', self.node_dict['step_loss'].tensor)
        weights_summary = tf.summary.histogram(
          self.vs_name + '/' + 'weights', self.node_dict['weights'].tensor)
        all_summaries += [
          clipped_grad_norm_summary, n_summary,
          step_loss_summary, seq_loss_summary, weights_summary]

      batch_size_summary = tf.summary.scalar(
        self.vs_name + '/' + 'batch_size', self.node_dict['batch_size'])
      all_summaries.append(batch_size_summary)

      if 'ent_reg' in self.node_dict:
        if avg_loss_by_n:
          ent_reg = (self.node_dict['ent_reg'] /
                     tf.cast(self.node_dict['n'],
                             dtype=tf.float32))
        else:
          ent_reg = self.node_dict['ent_reg'] / batch_size
        ent_reg_summary = tf.summary.scalar(
          self.vs_name + '/' + 'polic_entropy',
          (self.node_dict['ent_reg'] /
           tf.cast(self.node_dict['n'], tf.float32)))
        ent_reg_ppl_summary = tf.summary.scalar(
          self.vs_name + '/' + 'policy_entropy_ppl',
          tf.exp(self.node_dict['ent_reg'] /
                 tf.cast(self.node_dict['n'], tf.float32)))
        all_summaries.append(ent_reg_summary)
        all_summaries.append(ent_reg_ppl_summary)

    for s in self.node_dict['summaries']:
      all_summaries.append(s)

    merged = tf.summary.merge(inputs=all_summaries)
    self.node_dict.update(
      train=update, global_step=global_step,
      summaries=merged,
      update_lr=update_lr, new_lr=new_lr)

  @property
  def final_state(self):
    return 'final_state'

  @property
  def outputs(self):
    return 'outputs'

  @property
  def initial_state(self):
    return 'initial_state'

  @property
  def en_outputs(self):
    return 'en_outputs'

  @property
  def n_examples(self):
    return 'n_examples'

  @property
  def prediction_probs(self):
    return 'probs'

  @property
  def samples(self):
    return 'samples'

  @property
  def predictions(self):
    return 'predictions'

  @property
  def en_initial_state(self):
    return 'en_initial_state'

  def add_outputs(self, output_type, output_config):
    "Create part of the graph that compute final outputs from the RNN output."
    if output_type == 'softmax':
      self.add_softmax_outputs(**output_config)
    elif output_type == 'regression':
      self.add_regression_outputs(**output_config)
    else:
      raise NotImplementedError('Output type {} not supported!'.format(
        output_type))
    
  @with_graph_variable_scope
  def add_softmax_outputs(self, output_vocab_size=None, use_logits=None, name='Softmax'):
    "Add softmax layer on top of RNN outputs."
    with tf.variable_scope(name):
      seq_targets = create_seq_inputs(
        shape=tf.TensorShape([None, None]), dtype=tf.int32)
      seq_weights = create_seq_inputs(
        shape=tf.TensorShape([None, None]), dtype=tf.float32)      

      if use_logits:
        # Feeding logits instead of outputs (thus no linear transformation needed).
        logits, probs, predictions, samples, temperature = create_softmax_from_logits(
          self.node_dict['outputs'].tensor)
        sequence_length = self.node_dict['outputs'].sequence_length
      else:
        logits, probs, predictions, samples, temperature = create_softmax(
          self.node_dict['outputs'].tensor, output_vocab_size=output_vocab_size)
        sequence_length = self.node_dict['outputs'].sequence_length

      # From openai baselines to avoid numerical issue. 
      a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
      ea0 = tf.exp(a0)
      z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
      p0 = ea0 / z0

      clipped_entropy = p0 * (tf.log(z0) - a0)
      
      seq_entropy = SeqTensor(
        tf.reduce_sum(clipped_entropy, axis=-1) *
        tf.sequence_mask(
          sequence_length, dtype=tf.float32),
        sequence_length)

      policy_entropy = tf.reduce_sum(
        tf.reduce_sum(clipped_entropy, axis=-1) *
        tf.sequence_mask(
          sequence_length, dtype=tf.float32))

      # Compute sequence cross entropy loss.
      seq_logits, seq_probs, seq_predictions, seq_samples = [
        SeqTensor(x, sequence_length)
        for x in (logits, probs, predictions, samples)]
      xent_loss, sequence_loss, step_loss = create_seq_xent_loss(
        seq_logits.tensor,
        seq_targets.tensor,
        seq_weights.tensor,
        sequence_length)

    seq_step_loss = SeqTensor(step_loss, sequence_length)
      
    # Add new nodes to the node_dict.
    self.node_dict.update(
      targets=seq_targets, weights=seq_weights,
      temperature=temperature,
      sequence_loss=sequence_loss,
      step_loss=seq_step_loss,
      loss=xent_loss, 
      ent_reg=policy_entropy,
      seq_entropy=seq_entropy,
      probs=seq_probs,
      samples=seq_samples,
      predictions=seq_predictions, logits=seq_logits)


  @with_graph_variable_scope
  def add_regression_outputs(self, hidden_sizes=(), activation='relu',
                             stop_gradient=False, name='Regression'):
    "Add regression layer (linear transformation and MSE loss)."
    with tf.variable_scope(name):
      seq_targets = create_seq_inputs(
        shape=tf.TensorShape([None, None]), dtype=tf.float32)
      seq_weights = create_seq_inputs(
        shape=tf.TensorShape([None, None]), dtype=tf.float32)
      dropout = tf.placeholder_with_default(
        0.0, shape=None, name='regression_dropout')

      # Last dimension is the output, thus only one dimension.
      output_tensor = self.node_dict['outputs'].tensor
      if stop_gradient:
        output_tensor = tf.stop_gradient(output_tensor)
      if hidden_sizes:
          h = create_multilayer_fnn(
              output_tensor, dropout, list(hidden_sizes),
              activation=activation)
      else:
          h = output_tensor

      predictions = tf.layers.dense(inputs=h, units=1, name='regression_final')
      
      # Turn shape from (batch_size, timesteps, 1) to (batch_size, timesteps).
      predictions = tf.squeeze(predictions, axis=-1)
      sequence_length = self.node_dict['outputs'].sequence_length

      seq_predictions = SeqTensor(predictions, sequence_length)                         

      mse_loss = create_seq_mse_loss(
        seq_predictions.tensor, seq_targets.tensor, seq_weights.tensor, sequence_length)

    self.node_dict.update(
      targets=seq_targets, weights=seq_weights, regression_dropout=dropout,
      loss=mse_loss, predictions=seq_predictions)


class SeqGraph(Graph):
  "TensorFlow graph for RNN sequence model."  
  def __init__(self, graph_config, name='seq_graph'):
    super(SeqGraph, self).__init__(name)
    self.add_seq(**graph_config['core_config'])
    self.add_outputs(graph_config['output_type'], graph_config['output_config'])
    self.add_train(**graph_config['train_config'])
      
  @with_graph_variable_scope
  def add_seq(self, input_shape, input_vocab_size=None,
              hidden_size=128, n_layers=2,
              cell_type='lstm', bidirectional=False,
              dropout=0.0, use_embeddings=True,
              embedding_size=64, name='Sequence'):
    with tf.variable_scope(name):
      batch_size = tf.placeholder(
        dtype=tf.int32, shape=(), name='batch_size')
      if use_embeddings:
        embeddings = tf.get_variable(
          'embeddings', shape=(input_vocab_size, embedding_size),
          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
      else:
        embeddings = None        
      (seq_inputs, initial_state, seq_outputs, final_state,
       input_dropout, rnn_dropout, _) = create_seq_graph(
         input_shape, batch_size=batch_size,
         hidden_size=hidden_size, n_layers=n_layers,
         cell_type=cell_type, bidirectional=bidirectional,
         embeddings=embeddings)

      n = tf.reduce_sum(seq_inputs.sequence_length)

    self.node_dict.update(inputs=seq_inputs, 
                          rnn_dropout=rnn_dropout,
                          input_dropout=input_dropout,
                          embeddings=embeddings,
                          batch_size=batch_size,
                          final_state=final_state,
                          outputs=seq_outputs, n=n,
                          initial_state=initial_state)

class Seq2seqGraph(Graph):
  """TensorFlow graph for seq2seq model.

  A basic seq2seq model with attention. The model supports
  all the common specifications for a seq2seq model such as
  number of layers, whether to use bidirectional encoder,
  attention type, etc.

  """
  def __init__(self, graph_config, name='seq2seq_graph'):
    super(Seq2seqGraph, self).__init__(name)
    self.add_seq2seq(**graph_config['core_config'])
    self.add_outputs(graph_config['output_type'], graph_config['output_config'])
    self.add_train(**graph_config['train_config'])
  
  @with_graph_variable_scope
  def add_seq2seq(self, en_input_shape, input_shape,
                  use_attn=True,
                  attn_size=128, attn_vec_size=128,
                  en_input_vocab_size=None,
                  input_vocab_size=None,
                  en_hidden_size=128, en_n_layers=2,
                  hidden_size=128, n_layers=2,
                  cell_type='lstm',
                  en_bidirectional=False,
                  en_use_embeddings=True,
                  use_embeddings=True,
                  en_embedding_size=64,
                  embedding_size=64, name='Seq2seq'):
    with tf.variable_scope(name) as scope:
      batch_size = tf.placeholder(
        dtype=tf.int32, shape=[], name='batch_size')

      # Create encoder.
      with tf.variable_scope('Encoder'):
        if en_use_embeddings:
          en_embeddings = tf.get_variable(
            'embeddings', shape=(en_input_vocab_size, en_embedding_size),
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        else:
          en_embeddings = None        
        (en_seq_inputs, en_initial_state, en_seq_outputs, en_final_state,
         en_input_dropout, en_rnn_dropout, _) = create_seq_graph(
           en_input_shape, batch_size=batch_size,
           hidden_size=en_hidden_size, n_layers=en_n_layers,
           cell_type=cell_type, bidirectional=en_bidirectional,
           embeddings=en_embeddings, output_proj_size=en_hidden_size)

      if use_attn:
        attn_inputs = en_seq_outputs.tensor
      else:
        attn_inputs = None

      if en_bidirectional:
        en_final_state = en_final_state[0]

      # Create decoder.
      with tf.variable_scope('Decoder'):
        if use_embeddings:
          embeddings = tf.get_variable(
            'embeddings', shape=(input_vocab_size, embedding_size),
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        else:
          embeddings = None        
        (seq_inputs, initial_state, seq_outputs, final_state,
         input_dropout, rnn_dropout, _) = create_seq_graph(
           input_shape, batch_size=batch_size,
           hidden_size=hidden_size, n_layers=n_layers,
           cell_type=cell_type, bidirectional=False,
           embeddings=embeddings,
           attn_size=attn_size,
           attn_vec_size=attn_vec_size,
           attn_inputs=attn_inputs,
           initial_state=en_final_state)

      # Count number of steps.
      n = tf.reduce_sum(seq_inputs.sequence_length)

    self.node_dict.update(
      en_inputs=en_seq_inputs,
      en_rnn_dropout=en_rnn_dropout,
      en_input_dropout=en_input_dropout,
      en_outputs=en_seq_outputs,
      en_initial_state=en_initial_state,
      en_final_state=en_final_state,
      inputs=seq_inputs,
      rnn_dropout=rnn_dropout,
      input_dropout=input_dropout,
      outputs=seq_outputs,
      batch_size=batch_size,
      final_state=final_state,
      initial_state=initial_state, n=n,
      encoded_context=en_seq_outputs, context=en_seq_inputs,
      en_embeddings=en_embeddings,
      embeddings=embeddings)

    if use_attn:
      self.node_dict['attn_inputs'] = attn_inputs


class MemorySeq2seqGraph(Graph):
  def __init__(self, graph_config, name='memory_seq2seq_graph'):
    super(MemorySeq2seqGraph, self).__init__(name)
    self.use_gpu = graph_config['use_gpu']
    if self.use_gpu:
      os.environ["CUDA_VISIBLE_DEVICES"] = graph_config['gpu_id']
    else:
      os.environ["CUDA_VISIBLE_DEVICES"] = ''

    self.add_memory_seq2seq(**graph_config['core_config'])
    self.add_outputs(graph_config['output_type'], graph_config['output_config'])
    self.add_train(**graph_config['train_config'])
    self.config = graph_config
    
  @with_graph_variable_scope
  def add_memory_seq2seq(
      self, max_n_valid_indices=None,
      n_mem=None,
      n_builtin=None, 
      use_attn=True,
      attn_size=128, attn_vec_size=128,
      en_input_vocab_size=None,
      input_vocab_size=None,
      en_hidden_size=128, en_n_layers=2,
      hidden_size=128, n_layers=2,
      cell_type='lstm',
      en_bidirectional=False,
      en_use_embeddings=True,
      en_embedding_size=4,
      value_embedding_size=128,
      en_pretrained_vocab_size=None,
      en_pretrained_embedding_size=-1,
      tie_en_embeddings=True,
      add_lm_loss=False,
      n_en_input_features=1,
      n_de_output_features=1,
      en_attn_on_constants=False,
      name='MemorySeq2seq'):
    """Create seq2seq with key variable memory.

    Seq2seq with key variable memory is used for semantic
    parsing (generating programs from natural language
    instructions/questions).

    A MemorySeq2seq Model uses a memory cell in decoder.
    
    There are 3 types of tokens in a program:

    1) constants that are provided at the before the program
    is generated (added before decoding, different for
    different examples); 2) variables that saves the results
    from executing past expressions (added during decoding,
    different for different examples); 3) language
    primitives such as built-in functions and reserved
    tokens (for example, "(", ")"). (the same for different
    examples).

    There are two kinds of constants: 1) constants from the
    question, whose representation is from the span the
    annotated constants; 2) constants from the context,
    whose representation is from the constant value
    embeddings, for example, table columns. 
    
    So the decoder vocab is organized as
    [primitives, constants, variables].
    
    For a constant, its embedding is computed as sum of two
    parts: 1) embedding of the span (from encoder) on which
    the constant is annotated with, for example the span
    "barack obama" in "who is barack obama's wife" or the
    span "one" in "what is one plus one"; 2) embedding of
    the constant, for example, the embedding of the entity
    Obama or the embedding of the number one.

    For a variable, its embedding is the decoder RNN output
    at the step where the variable is created.

    For a primitive, its embedding is initialized randomly
    and tuned by SGD.

    Inspired by the code asistance (such as autocompletion)
    in modern IDE, we also apply semantic and syntax
    constraint on the decoder vocabulary so that at each
    step, only some of the tokens are valid. So the decoder
    has a dynamic vocabulary that is changing through
    different steps.

    """
    input_shape = tf_utils.MemoryInputTuple(
      tf.TensorShape([None, None]),
      tf.TensorShape([None, None]),
      tf.TensorShape([None, None, max_n_valid_indices]))
    
    input_dtype=tf_utils.MemoryInputTuple(
      tf.int32, tf.int32, tf.int32)

    en_input_shape=tf.TensorShape([None, None])
    
    constant_span_shape=tf.TensorShape([None, n_mem, 2])
    constant_value_embedding_shape=tf.TensorShape(
      [None, n_mem, value_embedding_size])
    builtin_de_embeddings_shape=tf.TensorShape([n_builtin, hidden_size])

    with tf.variable_scope('Constant_encoder'):
      # constant_span_embedding encodes the information
      # from the span where the constant is referred to,
      # for example the span "obama" in "who is the wife
      # of obama".

      # constant_value_embedding encodes the information
      # from the value of the constant, for example, the
      # embedding of the entity Obama.

      # constant_span: (B, n_mem, 2)
      constant_spans_placeholder = tf.placeholder(tf.int32, constant_span_shape)
      constant_spans = constant_spans_placeholder
      n_constants_placeholder = tf.placeholder(tf.int32, [None, 1])
      n_constants = tf.squeeze(n_constants_placeholder, [-1])

      # constant_spans: (B, n_mem, 1)
      # 0.0 if the span is [-1, -1], else 1.0.
      constant_span_masks = tf.cast(
        tf.greater(
          tf.reduce_sum(constant_spans, axis=2), 0), tf.float32)
      constant_span_masks = tf.expand_dims(constant_span_masks, -1)

      # constant_spans: (B, n_mem, 2, 1)
      constant_spans = tf.maximum(constant_spans, 0)
      constant_spans = tf.expand_dims(constant_spans, axis=-1)

      if constant_value_embedding_shape is not None:
        constant_value_embeddings_placeholder = tf.placeholder(
          tf.float32, shape=constant_value_embedding_shape)
        constant_value_embeddings = constant_value_embeddings_placeholder
        constant_value_embeddings = tf.layers.dense(
          constant_value_embeddings, hidden_size, use_bias=True)
        constant_value_masks = tf.squeeze(1 - constant_span_masks, [-1])

    if n_en_input_features > 0:
      en_input_features_shape = tf.TensorShape([None, None, n_en_input_features])
    else:
      en_input_features_shape = None

    with tf.variable_scope(name) as scope:
      batch_size = tf.placeholder(
        dtype=tf.int32, shape=[], name='batch_size')
      with tf.variable_scope('Encoder'):
        if en_use_embeddings:
          if en_pretrained_embedding_size < 0:
            en_embeddings = tf.get_variable(
              'embeddings', shape=(en_input_vocab_size, en_embedding_size),
              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
          else:
            en_embeddings = tf.get_variable(
              'embeddings', shape=(
                en_input_vocab_size - en_pretrained_vocab_size, en_embedding_size),
              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            en_pretrained_embeddings = tf.get_variable(
              'pretrained_embeddings', shape=(
                en_pretrained_vocab_size, en_pretrained_embedding_size),
              trainable=False,
              initializer=tf.zeros_initializer())
            en_pretrained_embeddings_placeholder = tf.placeholder(
              tf.float32, [en_pretrained_vocab_size, en_pretrained_embedding_size])
            en_pretrained_embeddings_init = en_pretrained_embeddings.assign(
              en_pretrained_embeddings_placeholder)
            en_pretrained_embeddings = tf.layers.dense(
              inputs=en_pretrained_embeddings, units=en_embedding_size,
              use_bias=True)
            en_embeddings = tf.concat(
              values=[en_embeddings, en_pretrained_embeddings], axis=0)
        else:
          en_embeddings = None

        if en_attn_on_constants: 
          tf.logging.info('Using attention in encoder!!!')
          (en_seq_inputs, en_initial_state, en_seq_outputs, en_final_state,
           en_input_dropout, en_rnn_dropout, en_rnn_outputs) = create_seq_graph(
             en_input_shape, batch_size=batch_size,
             hidden_size=en_hidden_size, n_layers=en_n_layers,
             cell_type=cell_type, bidirectional=en_bidirectional,
             embeddings=en_embeddings, output_proj_size=en_hidden_size,
             input_features_shape=en_input_features_shape,
             attn_inputs=constant_value_embeddings,
             attn_masks=constant_value_masks,
             attn_size=attn_size, attn_vec_size=attn_vec_size)
        else:
          (en_seq_inputs, en_initial_state, en_seq_outputs, en_final_state,
           en_input_dropout, en_rnn_dropout, en_rnn_outputs) = create_seq_graph(
             en_input_shape, batch_size=batch_size,
             hidden_size=en_hidden_size, n_layers=en_n_layers,
             cell_type=cell_type, bidirectional=en_bidirectional,
             embeddings=en_embeddings, output_proj_size=en_hidden_size,
             input_features_shape=en_input_features_shape)

        if n_en_input_features > 0:
          en_seq_input_features = SeqTensor(
            en_seq_inputs.tensor[1], tf.placeholder(tf.int32, [None]))
          en_seq_inputs = SeqTensor(
            en_seq_inputs.tensor[0], en_seq_inputs.sequence_length)
      
      if add_lm_loss:
        sequence_length = tf.maximum(en_seq_inputs.sequence_length - 1, 0)
        en_n = tf.cast(tf.reduce_sum(sequence_length), tf.float32)
        mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
        if en_bidirectional:
          en_fw_outputs = en_rnn_outputs[0]
          en_bw_outputs = en_rnn_outputs[1]

          if tie_en_embeddings:
            en_fw_logits = tf_utils.tensormul(
              en_fw_outputs[:, :-1, :], tf.transpose(en_embeddings))
            en_bw_logits = tf_utils.tensormul(
              en_bw_outputs[:, 1:, :], tf.transpose(en_embeddings))
          else:
            # Use 0 to n-2 to compute logits.
            en_fw_logits = tf.layers.dense(
              en_fw_outputs[:, :-1, :], en_input_vocab_size, use_bias=True)
            en_bw_logits = tf.layers.dense(
              en_bw_outputs[:, 1:, :], en_input_vocab_size, use_bias=True)
          
          # Use 1 to n-1 as targets.
          en_fw_lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=en_seq_inputs.tensor[:, 1:], logits=en_fw_logits) * mask
          en_bw_lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=en_seq_inputs.tensor[:, :-1], logits=en_bw_logits) * mask
          
          en_lm_loss = tf.reduce_sum(en_fw_lm_loss + en_bw_lm_loss) / en_n
        else:
          en_fw_outputs = en_rnn_outputs
          if tie_en_embeddings:
            en_fw_logits = tf_utils.tensormul(
              en_fw_outputs[:, :-1, :], tf.transpose(en_embeddings))
          else:
            en_fw_logits = tf.layers.dense(
              en_fw_outputs[:, :-1, :], en_input_vocab_size, use_bias=True)
          en_fw_lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=en_seq_inputs.tensor[:, 1:], logits=en_fw_logits) * mask
          en_lm_step_loss = en_fw_lm_loss
          en_lm_loss = tf.reduce_sum(en_lm_step_loss) / en_n
        
      if use_attn:
        attn_inputs = en_seq_outputs.tensor
        attn_masks = tf.sequence_mask(
          en_seq_outputs.sequence_length, dtype=tf.float32)
      else:
        attn_inputs = None
        attn_masks = None

      with tf.variable_scope('Constant_encoder'):
        batch_ind = tf.range(batch_size)

        # batch_ind: (B, 1, 1, 1)
        for i in range(3):
          batch_ind = tf.expand_dims(batch_ind, axis=-1)
          
        # batch_ind: (B, n_mem, 2, 1)
        batch_ind = tf.tile(batch_ind, [1, n_mem, 2, 1])

        # constant_span: (B, n_mem, 2, 2)
        constant_spans = tf.concat([batch_ind, constant_spans], axis=-1)
        
        # constant_span_embedding: (B, n_mem, 2, en_output_size)
        constant_span_embeddings = tf.gather_nd(en_seq_outputs.tensor, constant_spans)

        # constant_embedding: (B, n_mem, en_output_size)
        constant_embeddings = tf.reduce_mean(constant_span_embeddings, axis=2)
        constant_embeddings = constant_embeddings * constant_span_masks

        if constant_value_embedding_shape is not None:
          constant_embeddings = constant_embeddings + constant_value_embeddings
          
        # mask out the bad constants.
        # constant mask: (B, n_mem)
        constant_masks = tf.sequence_mask(
          n_constants, maxlen=n_mem, dtype=tf.float32)
        # constant mask: (B, n_mem, 1)
        constant_masks = tf.expand_dims(constant_masks, -1)
        constant_masks = tf.tile(constant_masks, [1, 1, hidden_size])
        # constant_embeddings: (B, n_mem, hidden_size)
        constant_embeddings = constant_embeddings * constant_masks

        # builtin_de_embeddings: (n_builtin, embed_size)
        builtin_de_embeddings = tf.get_variable(
          'builtin_de_embeddings', builtin_de_embeddings_shape,
          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        # builtin_de_embeddings: (1, n_builtin, embed_size)
        builtin_de_embeddings = tf.expand_dims(builtin_de_embeddings, axis=0) 
        # builtin_de_embeddings: (B, n_builtin, embed_size)
        builtin_de_embeddings = tf.tile(
          builtin_de_embeddings, [batch_size] + [1] * 2)

        # initial_memory: (B, n_builtin + n_mem, embed_size)
        initial_memory = tf.concat(
          [builtin_de_embeddings, constant_embeddings], axis=1)

        # concatenate static and constant embeddings to form
        # new memory to create initial states.
        if en_bidirectional:
          initial_state = en_final_state[0]
        else:
          initial_state = en_final_state

      with tf.variable_scope('Decoder'):
        initial_state = tf_utils.MemoryStateTuple(
          initial_memory, initial_state)  
        
        seq_inputs = create_seq_inputs(shape=input_shape, dtype=input_dtype)
        inputs = seq_inputs.tensor
        sequence_length = seq_inputs.sequence_length

        rnn_dropout = tf.placeholder_with_default(
          0.0, shape=None, name='rnn_dropout')        
        
        # Create multilayer attention cell then wrap with memory cell.
        cell = multilayer_dropout_cell(
          cell_fn=RNN_CELL_DICT[cell_type], hidden_size=hidden_size,
          n_layers=n_layers, dropout=rnn_dropout)
        
        if attn_inputs is not None:
          cell = tf_utils.SeqAttentionCellWrapper(
            cell, attn_inputs=attn_inputs,
            attn_size=attn_size, attn_vec_size=attn_vec_size,
            output_size=hidden_size, attn_masks=attn_masks)

        mem_size = builtin_de_embeddings_shape[0] + constant_span_shape[1]
        embed_size = hidden_size
        cell = tf_utils.MemoryWrapper(
          cell, mem_size, embed_size, max_n_valid_indices)

        flat_inputs = data_utils.flatten(inputs)
        flat_inputs = [
          tf.expand_dims(in_, -1) for in_ in flat_inputs[:2]] + flat_inputs[2:]
        inputs = data_utils.pack_sequence_as(inputs, flat_inputs)
        outputs, final_state = tf.nn.dynamic_rnn(
          cell, inputs, sequence_length=sequence_length,
          initial_state=initial_state, dtype=tf.float32)

        if n_de_output_features > 0:
          de_seq_output_features = create_seq_inputs(
            shape=tf.TensorShape(
              [None, None, max_n_valid_indices, n_de_output_features]),
            dtype=tf.int32, name='de_output_features')
          output_feature_weights = tf.get_variable(
            'de_output_feature_weights',
            shape=tf.TensorShape([n_de_output_features, 1]),
            initializer=tf.zeros_initializer())
          outputs = outputs + tf.squeeze(tf_utils.tensormul(
            tf.cast(de_seq_output_features.tensor, tf.float32),
            output_feature_weights), axis=-1)

        seq_outputs = SeqTensor(outputs, sequence_length)
            
      n = tf.reduce_sum(seq_inputs.sequence_length)

    self.node_dict.update(
      en_inputs=en_seq_inputs,
      en_rnn_dropout=en_rnn_dropout,
      en_input_dropout=en_input_dropout,
      en_outputs=en_seq_outputs,
      en_initial_state=en_initial_state,
      en_final_state=en_final_state,      
      inputs=seq_inputs,
      constant_spans=constant_spans_placeholder,
      constant_embeddings=constant_embeddings,
      constant_masks=constant_masks,
      n_constants=n_constants_placeholder,
      rnn_dropout=rnn_dropout,
      # input_dropout=input_dropout,
      outputs=seq_outputs,
      batch_size=batch_size,
      final_state=final_state,
      initial_state=initial_state, n=n,
      encoded_context=en_seq_outputs,
      context=en_seq_inputs,
      en_embeddings=en_embeddings)

    if en_pretrained_embedding_size > 0:
      self.node_dict['en_pretrained_embeddings'] = en_pretrained_embeddings_placeholder
      self.node_dict['en_pretrained_embeddings_init'] = en_pretrained_embeddings_init
  
    if constant_value_embedding_shape is not None:
      self.node_dict['constant_value_embeddings'] = constant_value_embeddings_placeholder

    if add_lm_loss:
      self.node_dict['en_lm_loss'] = en_lm_loss
      # self.node_dict['en_lm_step_loss'] = en_lm_step_loss

    if use_attn:
      self.node_dict['attn_inputs'] = attn_inputs

    if n_en_input_features > 0:
      self.node_dict['en_input_features'] = en_seq_input_features
      self.node_dict['summaries'].append(
        tf.summary.scalar(
          self.vs_name + '/' + 'en_input_features_sum',
          tf.reduce_sum(en_seq_input_features.tensor)))

    if n_de_output_features > 0:
      self.node_dict['output_features'] = de_seq_output_features
      self.node_dict['output_feature_weights'] = output_feature_weights
      self.node_dict['summaries'].append(
        tf.summary.scalar(
          self.vs_name + '/' + 'output_feature_weights_0',
          output_feature_weights[0][0]))
      self.node_dict['summaries'].append(
        tf.summary.scalar(
          self.vs_name + '/' + 'output_features_sum',
          tf.reduce_sum(de_seq_output_features.tensor)))


class MonitorGraph(object):
  """A tensorflow graph to monitor some values during training.

  Generate tensorflow summaries for the values to monitor
  them through tensorboard.
  """
  def __init__(self):
    self.node_dict = {}
    self._graph = tf.Graph()

  def launch(self):
    with self._graph.as_default():
      self.merged = tf.summary.merge_all()
      init = tf.global_variables_initializer()
    self.session = tf.Session(graph=self._graph)
    self.session.run(init)

  def add_scalar_monitor(self, name, dtype):
    with self._graph.as_default():
      x = tf.placeholder(
        dtype=dtype, shape=None, name=name)
      tf.summary.scalar(name, x)
    self.node_dict[name] = x

  def generate_summary(self, feed_dict):
    summary_str = self.session.run(
      self.merged, map_dict(self.node_dict, feed_dict))
    return summary_str
      

# Utility functions for creating TensorFlow graphs.

# FNN
def create_multilayer_fnn(inputs, dropout, hidden_sizes, activation='relu'):
  x = inputs
  for size in hidden_sizes:
    x = tf.nn.dropout(x, 1 - dropout)
    x = tf.layers.dense(inputs=x, units=size, activation=ACTIVATION_DICT[activation])
  return x


# Loss 
def create_seq_mse_loss(outputs, targets, weights, sequence_length):
  mask = tf.sequence_mask(
    sequence_length, dtype=tf.float32)
  loss = tf.reduce_sum(tf.squared_difference(outputs, targets) * weights * mask)
  return loss


def create_seq_xent_loss(logits, targets, weights, sequence_length):
  mask = tf.sequence_mask(
    sequence_length, maxlen=tf.reduce_max(sequence_length),
    dtype=tf.float32)
  step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=targets, logits=logits) * weights * mask
  sequence_loss = tf.reduce_sum(step_loss, axis=1)
  xent_loss = tf.reduce_sum(sequence_loss)
  return xent_loss, sequence_loss, step_loss


def create_softmax(
    inputs, softmax_w=None,
    output_vocab_size=None, use_bias=False, name='Softmax_layer'):
  "Create nodes for linear transformation of inputs and softmax computation."
  with tf.name_scope(name):
    # inputs = tf.nn.dropout(inputs, 1-dropout)
    if softmax_w is None:
      logits = tf.layers.dense(
        inputs=inputs, units=output_vocab_size,
        use_bias=use_bias)
    else:
      logits = tf_utils.tensormul(inputs, softmax_w)
      if use_bias:
        softmax_b = tf.Variable(
          initial_value=np.zeros(
            (1, output_vocab_size), dtype=tf.float32),
          name='softmax_bias')
        logits += softmax_b
    return create_softmax_from_logits(logits)


def create_softmax_from_logits(logits):
  "Create nodes for softmax computation from logits."
  temperature = tf.placeholder_with_default(
    1.0, shape=None, name='temperature')
  logits = logits / temperature

  logits_shape = tf.shape(logits)
  logits_dim = logits_shape[-1]
  logits_2d = tf.reshape(logits, [-1, logits_dim])
  samples = tf.multinomial(logits_2d, 1)
  samples = tf.reshape(samples, logits_shape[:-1])

  probs = tf.nn.softmax(logits)
  predictions = tf.argmax(probs, axis=2)
    
  return logits, probs, predictions, samples, temperature


# Embedding
def embed_inputs(
    inputs, embeddings, name='Embedding_layer'):
  with tf.name_scope(name):
    embedded_inputs = tf.nn.embedding_lookup(embeddings, inputs)
    return embedded_inputs


# RNN
def create_rnn(
    cell, initial_state, inputs, sequence_length,
    hidden_size, bidirectional, cell_bw=None,
    name='RNN'):
  with tf.name_scope(name):
    if bidirectional:
      # Note that you can't use bidirectional RNN if you
      # want to do decoding.
      initial_state_fw = initial_state[0]
      initial_state_bw = initial_state[1]
      outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
        cell, cell_bw, inputs, sequence_length=sequence_length,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw,
        dtype=tf.float32)
    else:
      outputs, final_state = tf.nn.dynamic_rnn(
        cell, inputs, sequence_length=sequence_length,
        initial_state=initial_state, dtype=tf.float32)
  return outputs, final_state

    
# RNN Cell
def multilayer_dropout_cell(
    cell_fn, hidden_size, n_layers, dropout,
    use_skip_connection=True):
  """Create multilayer RNN cell with dropout."""
  cells = []
  for i in xrange(n_layers):
    cell = cell_fn(hidden_size)
    if i > 0 and use_skip_connection:
      cell = tf.nn.rnn_cell.ResidualWrapper(cell)
    cell = tf.contrib.rnn.DropoutWrapper(
      cell, output_keep_prob=1.0-dropout)
    # variational_recurrent=True,
    # state_keep_prob = 1.0 - dropout,
    # dtype=tf.float32)
    cells.append(cell)

  final_cell = tf.contrib.rnn.MultiRNNCell(cells)
  return final_cell


# Input placeholders.
def create_seq_inputs(shape, dtype=tf.float32, name='inputs'):
  with tf.name_scope(name):
    if isinstance(shape, tuple):
      flat_input_shape = data_utils.flatten(shape)
      assert isinstance(dtype, tuple)
      flat_dtype = data_utils.flatten(dtype)
      flat_inputs = [tf.placeholder(
        dt, sh, name='inputs') for dt, sh in zip(flat_dtype, flat_input_shape)]
      inputs = data_utils.pack_sequence_as(shape, flat_inputs)
    else:
      inputs = tf.placeholder(dtype, shape)
    sequence_length = tf.placeholder(
      tf.int32, [None], name='sequence_length')
  return SeqTensor(inputs, sequence_length)
  
def create_tuple_placeholders_with_default(inputs, extra_dims, shape):
  if isinstance(shape, int):
    result = tf.placeholder_with_default(
      inputs, list(extra_dims) + [shape])
  else:
    subplaceholders = [create_tuple_placeholders_with_default(
      subinputs, extra_dims, subshape)
                       for subinputs, subshape in zip(inputs, shape)]
    t = type(shape)
    if t == tuple:
      result = t(subplaceholders)
    else:
      result = t(*subplaceholders)    
  return result

        
def create_tuple_placeholders(dtype, extra_dims, shape):
  if isinstance(shape, int):
    result = tf.placeholder(dtype, list(extra_dims) + [shape])
  else:
    subplaceholders = [create_tuple_placeholders(dtype, extra_dims, subshape)
                       for subshape in shape]
    t = type(shape)

    # Handles both tuple and LSTMStateTuple.
    if t == tuple:
      result = t(subplaceholders)
    else:
      result = t(*subplaceholders)
  return result


# Sequence models.
def create_seq_graph(input_shape, batch_size=None, 
                     # input_vocab_size=None,
                     attn_inputs=None,
                     attn_size=128, attn_vec_size=128,
                     # output_size=128,
                     input_size=None, 
                     hidden_size=128, n_layers=2,
                     cell_type='lstm', bidirectional=False,
                     initial_state=None,
                     #use_embeddings=True,
                     #embedding_size=64,
                     embeddings=None,
                     output_proj_size=None,
                     input_features_shape=None,
                     attn_masks=None):
  # Create inputs.
  seq_inputs = create_seq_inputs(
    shape=input_shape, dtype=tf.int32 if embeddings is not None else tf.float32)

  rnn_dropout = tf.placeholder_with_default(
    0.0, shape=None, name='rnn_dropout')

  # Create embedding layer.
  if embeddings is not None:
    embedded_inputs = embed_inputs(
      seq_inputs.tensor, embeddings=embeddings)
  else:
    embedded_inputs = seq_inputs.tensor
    
  input_dropout = tf.placeholder_with_default(
    0.0, shape=None, name='input_dropout')
    
  embedded_inputs = tf.nn.dropout(
    embedded_inputs, 1 - input_dropout)

  # If we include features in inputs, then add them here.
  if input_features_shape is not None:
    seq_input_features = create_seq_inputs(
      shape=input_features_shape, dtype=tf.int32)
    embedded_inputs = tf.concat(
      [embedded_inputs, tf.cast(seq_input_features.tensor, tf.float32)],
      axis=-1)
    seq_inputs = SeqTensor(
      (seq_inputs.tensor, seq_input_features.tensor), seq_inputs.sequence_length)
  else:
    seq_input_features = None
  
  embedded_seq_inputs = SeqTensor(embedded_inputs, seq_inputs.sequence_length)
    
  # Create RNN cell
  cell = multilayer_dropout_cell(
    RNN_CELL_DICT[cell_type],
    hidden_size, n_layers, rnn_dropout)

  if bidirectional:
    cell_bw = multilayer_dropout_cell(
      RNN_CELL_DICT[cell_type],
      hidden_size, n_layers, rnn_dropout)
  else:
    cell_bw = None

  # Add attention.
  if attn_inputs is not None:
    cell = tf_utils.SeqAttentionCellWrapper(
      cell, attn_inputs=attn_inputs,
      attn_size=attn_size, attn_vec_size=attn_vec_size,
      output_size=hidden_size, attn_masks=attn_masks)
    if bidirectional:
      cell_bw = tf_utils.SeqAttentionCellWrapper(
        cell_bw, attn_inputs=attn_inputs,
        attn_size=attn_size, attn_vec_size=attn_vec_size,
        output_size=hidden_size, attn_masks=attn_masks)
    
  if initial_state is None:
    # Create zero state.
    zero_state = cell.zero_state(batch_size, tf.float32)
    
    if bidirectional:
      zero_state_bw = cell_bw.zero_state(batch_size, tf.float32)
      zero_state = (zero_state, zero_state_bw)

    initial_state = zero_state
      
  # Create RNN.
  outputs, final_state = create_rnn(
    cell, initial_state, embedded_seq_inputs.tensor,
    embedded_seq_inputs.sequence_length,
    hidden_size=hidden_size,
    bidirectional=bidirectional, cell_bw=cell_bw)
  rnn_outputs = outputs

  if bidirectional:
    outputs = tf.concat(outputs, axis=2)
    hidden_size *= 2
    
  # Whether to add linear transformation to outputs. 
  if output_proj_size is not None:
    outputs = tf.layers.dense(
      inputs=outputs, units=output_proj_size, use_bias=True)

  seq_outputs = SeqTensor(
    outputs,
    tf.placeholder_with_default(
      seq_inputs.sequence_length, shape=[None]))
  
  return (seq_inputs, initial_state, seq_outputs,
          final_state, input_dropout, rnn_dropout,
          rnn_outputs)


# General utility functions.

def map_dict(dict_1, main_dict):
  new_dict = {}  
  for k, v in main_dict.iteritems():
    if k in dict_1:
      new_dict[dict_1[k]] = main_dict[k]
  return new_dict
