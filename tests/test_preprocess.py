from ..src.features.preprocess import *

sentence = ['Sign in to comment!, With about 900,000 concealed handgun permit holders in Texas, ']


def test_remove_hyperlink():
    sentence = "www.hyperlinnk.com adresinde"
    predicted = remove_hyperlink(sentence)
    expected = "adresinde"
    assert predicted == expected


def test_tokenize():
    sentence = "Bu cümle çok uzun değil ya."
    predicted = tokenize_sentence(sentence)
    tester = ['Bu', 'cümle', 'çok', 'uzun', 'değil', 'ya', '.']
    assert predicted == tester


def test_preprocess_operations():
    preprocess_operations = [to_lower, remove_hyperlink, remove_number, remove_punctuation, remove_whitespace,
                             replace_special_chars, remove_stopwords, apply_stemmer, remove_less_than_two]
    predicted = apply_preprocess_operations_to_corpus(sentence, preprocess_operations)
    tester = ['sign comment conceal handgun permit holder texa']
    assert predicted == tester
