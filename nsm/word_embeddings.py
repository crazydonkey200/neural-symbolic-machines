import json
import numpy as np
import gensim


class EmbeddingModel(object):
  def __init__(
      self, vocab_file, embedding_file, normalize_embeddings=True):
    with open(embedding_file, 'rb') as f:
      self.embedding_mat = np.load(f)
    if normalize_embeddings:
      self.embedding_mat = self.embedding_mat / np.linalg.norm(
        self.embedding_mat, axis=1, keepdims=True)
    with open(vocab_file, 'r') as f:
      tks = json.load(f)
    self.vocab = dict(zip(tks, range(len(tks))))

  def __contains__(self, word):
    return word in self.vocab
    
  def __getitem__(self, word):
    if word in self.vocab:
      index = self.vocab[word]
      return self.embedding_mat[index]
    else:
      raise KeyError
