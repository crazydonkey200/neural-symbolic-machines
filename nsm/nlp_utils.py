"""NLP utility functions."""
import nltk


def tokenize(string):
  return nltk.word_tokenize(string)


def bleu_score(sentence, gold_sentence):
  return nltk.translate.bleu_score.sentence_bleu(
      [gold_sentence], sentence)


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
  return nltk.metrics.distance.edit_distance(
      s1, s2, substitution_cost=substitution_cost,
      transpositions=transpositions)
