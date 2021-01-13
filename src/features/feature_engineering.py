from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings

warnings.filterwarnings(action='ignore')


class FeatureEngineering:
    """
    This class applies text feature engineering operations: tf-idf, cv or word2vec
    """

    def __init__(self, model="tf_idf", analizer="word", n_gram=(1, 1)):
        self.tf_idf = TfidfVectorizer(max_df=0.5, min_df=2, analyzer=analizer, ngram_range=n_gram)
        self.cv = CountVectorizer(analyzer=analizer, ngram_range=n_gram)
        self.model = model

    def get_frequency_based_model(self):
        if self.model == "tf_idf":
            feature_model = self.tf_idf
        elif self.model == "cv":
            feature_model = self.cv
        else:
            raise IOError("Wrong model feature engineering type selected!")
        return feature_model

    def fit_transform(self, corpus):
        return self.get_frequency_based_model().fit_transform(corpus)

    def fit(self, corpus):
        return self.get_frequency_based_model().fit(corpus)

    def transform(self, corpus):
        return self.get_frequency_based_model().transform(corpus)
