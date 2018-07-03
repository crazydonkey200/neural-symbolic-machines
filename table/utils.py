import re
import numpy as np

import babel
from babel import numbers
from wtq import evaluator


def average_token_embedding(tks, model, embedding_size=300):
  arrays = []
  for tk in tks:
    if tk in model:
      arrays.append(model[tk])
    else:
      arrays.append(np.zeros(embedding_size))
  return np.average(np.vstack(arrays), axis=0)


def get_embedding_for_constant(value, model, embedding_size=300):
  if isinstance(value, list):
    # Use zero embeddings for values from the question to
    # avoid overfitting
    return np.zeros(embedding_size)
  elif value[:2] == 'r.':
    value_split = value.split('-')
    type_str = value_split[-1]
    type_embedding = average_token_embedding([type_str], model)
    value_split = value_split[:-1]
    value = '-'.join(value_split)
    raw_tks = value[2:].split('_')
    tks = []
    for tk in raw_tks:
      valid_tks = find_tk_in_model(tk, model)
      tks += valid_tks
    val_embedding = average_token_embedding(tks or raw_tks, model)
    return (val_embedding + type_embedding) / 2
  else:
    raise NotImplementedError('Unexpected value: {}'.format(value))


def find_tk_in_model(tk, model):
    special_tk_dict = {'-lrb-': '(', '-rrb-': ')'}
    if tk in model:
        return [tk]
    elif tk in special_tk_dict:
        return [special_tk_dict[tk]]
    elif tk.upper() in model:
        return [tk.upper()]
    elif tk[:1].upper() + tk[1:] in model:
        return [tk[:1].upper() + tk[1:]]
    elif re.search('\\/', tk):
        tks = tk.split('\\\\/')
        if len(tks) == 1:
          return []
        valid_tks = []
        for tk in tks:
            valid_tk = find_tk_in_model(tk, model)
            if valid_tk:
                valid_tks += valid_tk
        return valid_tks
    else:
        return []


# WikiSQL evaluation utility functions.
def wikisql_normalize(val):
  """Normalize the val for wikisql experiments."""
  if (isinstance(val, float) or isinstance(val, int)):
    return val
  elif isinstance(val, str) or isinstance(val, unicode):
    try:
      val = babel.numbers.parse_decimal(val)
    except babel.numbers.NumberFormatError:
      val = val.lower()
    return val
  else:
    return None


def wikisql_process_answer(answer):
  processed_answer = []
  for a in answer:
    normalized_val = wikisql_normalize(a)
    # Ignore None value and normalize the rest, keep the
    # order.
    if normalized_val is not None:
      processed_answer.append(normalized_val)
  return processed_answer


def wikisql_score(prediction, answer):
  prediction = wikisql_process_answer(prediction)
  if prediction == answer:
    return 1.0
  else:
    return 0.0


# WikiTableQuestions evaluation function.
def wtq_score(prediction, answer):
    processed_answer = evaluator.target_values_map(*answer)
    correct = evaluator.check_prediction(
      [unicode(p) for p in prediction], processed_answer)
    if correct:
        return 1.0
    else:
        return 0.0    
