import re
import string
import nltk
from nltk import SnowballStemmer
import warnings
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

if False:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
warnings.filterwarnings(action='ignore')

wpt = nltk.WordPunctTokenizer()
PorterStemmer = PorterStemmer()
SnowballStemmer = SnowballStemmer(language="english")
lemmatizer = WordNetLemmatizer()


def remove_hyperlink(sentence: str) -> str:
    """
    This method remove hyperlinks & emails & mentions  from given sentence

    :param sentence: input sentence file, :type str
    :return:
    """
    sentence = re.sub(r"\S*@\S*\s?", " ", sentence)
    sentence = re.sub(r"www\S+", " ", sentence)
    sentence = re.sub(r"http\S+", " ", sentence)
    return sentence.strip()


def to_lower(sentence: str) -> str:
    """
    This method lowers sentence

    :param sentence: input sentence file, :type str
    :return:
    """
    result = sentence.lower()
    return result


def remove_number(sentence: str) -> str:
    """
    This method removes numbers from given sentence

    :param sentence: input sentence file, :type str
    :return:
    """
    result = re.sub(r'\S*\d\S*', ' ', sentence)
    return result


def remove_punctuation(sentence: str) -> str:
    """
    This method remove punctuations from given sentence

    :param sentence: input sentence file, :type str
    :return:
    """
    result = sentence.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result


def remove_whitespace(sentence: str) -> str:
    """
    This method removes extra white spaces from given sentence

    :param sentence: input sentence file, :type str
    :return:
    """
    result = sentence.strip()
    return result


def replace_special_chars(sentence: str) -> str:
    """
    This method replaces newline character with space

    :param sentence: input sentence file, :type str
    :return:
    """
    chars_to_remove = ['\t', '\n', ';', "!", '"', "#", "%", "&", "'", "(", ")",
                       "+", ",", "-", "/", ":", ";", "<",
                       "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                       "`", "{", "|", "}", "~", "–", '”', '“', '’']
    for ch in chars_to_remove:
        sentence = sentence.replace(ch, ' ')
    # replace ascii chars with symbol 8
    sentence = sentence.replace(u'\ufffd', ' ')
    return sentence.strip()


def remove_stopwords(sentence: str, stopwords: list = ENGLISH_STOP_WORDS) -> str:
    """
    This method removes stopwords from given sentence

    :param sentence: sentence to remove stopwords, :type str
    :param stopwords: stopwords list, :type list
    :return: cleaned sentence
    """
    tokens = sentence.split()
    filtered_tokens = [token for token in tokens if token not in stopwords]
    sentence = ' '.join(filtered_tokens)
    return sentence


def apply_stemmer(sentence: str, stemmer_name=PorterStemmer) -> str:
    """
    This method applies stemmer to given sentence

    :param sentence: input string, :type str
    :param stemmer_name: stemmer to apply: SnowballStemmer | PorterStemmer
    :return:
    """
    tokens = sentence.split()
    tokens = pos_tag(tokens)
    # don't apply proper names
    stemmed_tokens = [stemmer_name.stem(key.lower()) for key, value in tokens if value != 'NNP']
    sentence = ' '.join(stemmed_tokens)
    return sentence


def apply_lemmatizer(sentence: str) -> str:
    """
    This method applies lemma to given sentence

    :param sentence: sentence to apply lemma operation, :type str
    :return:
    """
    tokens = sentence.split()
    lemmatize_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    sentence = ' '.join(lemmatize_tokens)
    return sentence


def remove_less_than_two(sentence: str) -> str:
    """
    This method removes less than two chars from given sentence

    :param sentence: input sentence, :type str
    :return:
    """
    tokens = sentence.split()
    filtered_tokens = [token for token in tokens if len(token) > 2]
    sentence = ' '.join(filtered_tokens)
    return sentence


def tokenize_sentence(sentence: str) -> str:
    """
    This method tokenize sentences into tokens

    :param sentence: sentence to tokenize, :type str
    :return:
    """
    return wpt.tokenize(sentence)


def tokenize_list_of_sentences(sentences: list) -> list:
    """
    This method tokenize list of sentences

    :param sentences: sentence list
    :return:
    """
    return [tokenize_sentence(sentence=sentence) for sentence in sentences]


def basic_preprocess_operations(sentence: str) -> str:
    """
    This method applies basic preprocess operations to given sentence:
      remove_hyperlink & replace_newline & to_lower & remove_number & remove_punctuation & remove_whitespace

    :param sentence: sentence to apply preprocess operation, :type str
    :return:
    """
    cleaning_utils = [remove_hyperlink,
                      replace_special_chars,
                      to_lower,
                      remove_number,
                      remove_punctuation, remove_whitespace]
    for o in cleaning_utils:
        sentence = o(sentence)
    return sentence


def apply_preprocess_operations_to_corpus(corpus: list, operations: list, **kwargs) -> list:
    """
    This method applies list of operations to given corpus

    :param corpus: list of sentences, :type list
    :param operations: list of operations, :type list
       operations:
           - remove_less_than_two
           - apply_lemmatizer
           - apply_stemmer
           - remove_stopwords
           - replace_special_chars
           - remove_whitespace
           - remove_punctuation
           - remove_number
           - to_lower
           - remove_hyperlink
    :param kwargs:(optional) params to apply operations,
                  for stemmer stemmer operation and for remove stopwords stopwords list

    :return: preprocessed sentences, :type list
    """
    for operation in operations:
        if operation == apply_stemmer:
            if "stemmer_name" in kwargs:
                corpus = apply_operation(corpus, apply_stemmer, kwargs.get("stemmer_name"))
            else:
                corpus = apply_operation(corpus, apply_stemmer)
        elif operation == remove_stopwords:
            if "stopwords" in kwargs:
                corpus = apply_operation(corpus, remove_stopwords, kwargs.get("stopwords"))
            else:
                corpus = apply_operation(corpus, remove_stopwords)
        else:
            corpus = apply_operation(corpus, operation)
    return corpus


def apply_operation(corpus, operation, **kwargs):
    """
    This method applies one operation and returns the result

    :param corpus: list of sentences, :type list
    :param operation: image operation
    :param kwargs: (optional) params to apply operations,
                  for stemmer stemmer operation and for remove stopwords stopwords list
    :return operation applied result
    """
    data_precessed = []
    for sentence in corpus:
        data_precessed.append(operation(sentence, **kwargs))
    return data_precessed

