import pandas as pd
import streamlit as st
import re
import string
import time
import nltk
from nltk import SnowballStemmer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from pandas import DataFrame
from sklearn_extra.cluster import KMedoids
import collections
import pandas as pd
import warnings
import matplotlib.cm as cm

warnings.filterwarnings(action='ignore')
plt.style.use('ggplot')

if False:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

wpt = nltk.WordPunctTokenizer()
PorterStemmer = PorterStemmer()
SnowballStemmer = SnowballStemmer(language="english")
lemmatizer = WordNetLemmatizer()


def pipeline(corpus, n_cluster: int, labels: list, max_df: int = 1, min_df: int = 1,
             max_features: int = None, analyzer: str = "word", n_gram=(1, 1),
             n_components: int = 40, idf: bool = True):
    """
    This method applies tf-idf, svd and k-means clustering with given parameters
    :param n_gram:
    :param analyzer:
    :param labels: list of labels
    :param idf: use idf
    :param corpus: data to train
    :param n_cluster: number of clusters
    :param max_df:
    :param min_df:
    :param max_features:
    :param n_components:
    :return:
    """
    pkl_filename = f"model_{n_cluster}_{min_df}_{max_features}_{n_components}_{n_gram}.pkl"
    # apply tf-idf vectorizer
    vectorizer = TfidfVectorizer(max_df=max_df, max_features=max_features,
                                 min_df=min_df, use_idf=idf, analyzer=analyzer, ngram_range=n_gram)
    transformed = vectorizer.fit_transform(corpus)

    # apply dimention reduction
    svd = TruncatedSVD(n_components=n_components)
    vec_matrix_svd = svd.fit_transform(transformed)

    # apply k-mean
    km = KMeans(n_clusters=n_cluster, verbose=0)
    km.fit(vec_matrix_svd)
    y_ = km.predict(vec_matrix_svd)

    resulted_data = pd.DataFrame({'Completeness': [metrics.completeness_score(labels, km.labels_)],
                                  'Homogeneity': [metrics.homogeneity_score(labels, km.labels_)]})

    st.dataframe(resulted_data)

    return km, y_, vec_matrix_svd


def apply_svd(data: DataFrame, n_components: int) -> DataFrame:
    """
    This method applies truncated svd to in put data with given dimension
    :param data: input dataframe, :type DataFrame,
    :param n_components: Desired dimensionality of output data. , :type int
    :return DataFrame
    """
    svd = TruncatedSVD(n_components=n_components)
    return svd.fit_transform(data)


def get_model(model: str, data, **kwargs):
    """
    This  method returns selected clustering model
    """
    if model == 'k-means':
        cluster = KMeans(**kwargs)
    elif model == 'k-medoids':
        cluster = KMedoids(**kwargs)
    elif model == 'dbscan':
        cluster = DBSCAN(**kwargs)
    elif model == 'ahct':
        cluster = AgglomerativeClustering(**kwargs)
    elif model == 'mean-shift':
        bandwith = estimate_bandwidth(data)
        cluster = MeanShift(bandwidth=bandwith, **kwargs)
    else:
        st.write("Wrong clustering type selected !!")
        return None
    return cluster


def apply_tf_idf(data, max_df: int = 1, min_df: int = 1, idf: bool = True,
                 max_features: int = None, analyzer: str = "word", n_gram=(1, 1)):
    """
    This method applies tf-idf to given data
    see documentation: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    :param idf:
    :param data:
    :param max_df:
    :param min_df:
    :param max_features:
    :param analyzer:
    :param n_gram:
    :return:
    """
    vectorizer = TfidfVectorizer(max_df=max_df, max_features=max_features,
                                 min_df=min_df, use_idf=idf, analyzer=analyzer, ngram_range=n_gram)
    transformed = vectorizer.fit_transform(data)
    return transformed


def apply_svd_pipeline(data: DataFrame, n_components: int, model: str, max_df: int = 1, min_df: int = 1,
                       idf: bool = True, max_features: int = None, analyzer: str = "word", n_gram=(1, 1), **kwargs):
    """
    This method applies svd data dimension reduction method
    :param n_gram:
    :param analyzer:
    :param max_features:
    :param idf:
    :param min_df:
    :param max_df:
    :param data: DataFrame
    :param n_components: number of result dataframe features,, :type int
    :param model: selected clustering algorithm, :type str
         ['k-means', 'k-medoids', 'dbscan', 'ahct', 'mean-shift']
    :return model
    """
    pkl_filename = f"svd_{model}_{min_df}_{max_features}_{n_components}_{n_gram}.pkl"

    transformed = apply_tf_idf(data, max_df, min_df, idf, max_features, analyzer, n_gram)
    data_svd = apply_svd(transformed, n_components)
    st.write(f"Shape of the data:{data_svd.shape}")

    cluster = get_model(model, data, **kwargs)
    if cluster is None:
        st.write("Wrong clustering type selected !!")
        return
    y_ = cluster.fit_predict(data_svd)

    st.write(f"Number of samples in each cluster:{collections.Counter(y_)}")

    return cluster, y_, data_svd


def apply_clustering(data: DataFrame, model: str, max_df: int = 1, min_df: int = 1,
                     idf: bool = True, max_features: int = None, analyzer: str = "word", n_gram=(1, 1), **kwargs):
    """
    This method applies svd data dimension reduction method
    :param n_gram:
    :param analyzer:
    :param max_features:
    :param idf:
    :param min_df:
    :param max_df:
    :param data: DataFrame
    :param model: selected clustering algorithm, :type str
         ['k-means', 'k-medoids', 'dbscan', 'ahct', 'mean-shift']
    :return model
    """
    pkl_filename = f"{model}_{min_df}_{max_features}_{max_features}_{n_gram}.pkl"
    st.write(f"Shape of the data:{data.shape}")

    transformed = apply_tf_idf(data, max_df, min_df, idf, max_features, analyzer, n_gram)
    cluster = get_model(model, transformed, **kwargs)
    if cluster is None:
        st.write("Wrong clustering type selected !!")
        return
    y_ = cluster.fit_predict(transformed)

    st.write(f"Number of samples in each cluster:{collections.Counter(y_)}")

    return cluster, y_, transformed


def plot_elbow(data: DataFrame, num_of_cluster_range: set):
    """
    This method provides elbow ploting result

    :param data: input dataframe, :type DataFrame,
    :num_of_cluster_range: range of the clusters, :type str
    :return plot
    """
    wcss = list()
    for i in range(num_of_cluster_range[0], num_of_cluster_range[1]):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    fig, ax = plt.subplots(figsize=(24.0, 16.0))
    ax.plot(range(num_of_cluster_range[0], num_of_cluster_range[1]), wcss)
    ax.set_title('The Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('wcss')
    st.pyplot(fig)


def plot_cluster_result(data: DataFrame, predictions: list, model, title: str):
    """
    This method plot clustering results

    :param data: trained DataFrame,
    :param predictions: prediction results of model, :type list
    :param model: fitted clustering model
    :param title: title for plot
    """
    number_of_class = len(set(predictions))
    colors = cm.rainbow(np.linspace(0, 1, number_of_class))
    fig, ax = plt.subplots(figsize=(24.0, 16.0))
    for i in range(0, number_of_class):
        ax.scatter(data[predictions == i, 0], data[predictions == i, 1], s=100, c=colors[i], label=f'cluster{str(i)}')
    try:
        ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300, c='blue', label='Centroids')
    except:
        st.write("used model has no cluster centers")
    ax.set_title(title)
    st.pyplot(fig)


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


OPERATIONS = {"lower": to_lower, "remove hyperlink": remove_hyperlink, "remove number": remove_number,
              "remove punctuation": remove_punctuation,
              "remove whitespace": remove_whitespace,
              "replace special chars": replace_special_chars, "remove stopwords": remove_stopwords,
              "apply stemmer": apply_stemmer,
              "remove less than two": remove_less_than_two}

st.title('K-Means Clustering of Textual Data')

import base64
file_ = open("C:\\Users\\user\\Desktop\\nlp_project\\docs\\kmeans.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

uploaded_file = st.file_uploader("Choose a CSV file")
page_bg_img = '''
<img src="C:\\Users\\user\\Desktop\\nlp_project\\docs\\kmeans.gif">
<style>
body {
    color: #fff;
    background-color: #FFFFFF;
}
.stButton>button {
    color: #4F8BF9;
}

.stTextInput>div>div>input {
    color: #4F8BF9;
}
</style>
'''

st.markdown(f'<img src="data:image/gif;base64,{data_url}">', unsafe_allow_html=True)


def print_quality_result(dataframe):
    len_df = len(dataframe)
    columns = list(dataframe.columns)
    st.header('Data Quality Result')
    st.markdown(f'Number of sample in data set: **{len_df}**.')
    st.markdown(f'Columns in data set: **{columns}**.')
    if dataframe.info() is not None:
        st.subheader("Data Info")
        st.dataframe(dataframe.info())

    if dataframe.describe() is not None:
        st.subheader("Data Statistics")
        st.dataframe(dataframe.describe())

    percent_missing = dataframe.isnull().sum() * 100 / len(dataframe)
    missing_value_df = pd.DataFrame({'column_name': dataframe.columns,
                                     'percent_missing': percent_missing})
    st.subheader("DataFrame Null Percentages")
    st.dataframe(missing_value_df)


if uploaded_file is not None:
    st.write("Uploaded data frame:")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    if st.button('Create Data Quality Report'):
        print_quality_result(df)

    df_selected = df.dropna()
    df_selected = df_selected.drop_duplicates()
    st.header("Preprocess Operations")
    column = st.selectbox("Select column to apply preprocess operation:",
                          df.columns)
    if df[column].dtype == np.object:
        operations = st.multiselect("Select operations to apply:",
                                    ('lower', 'remove hyperlink', "remove number", "remove punctuation",
                                     "remove whitespace",
                                     "replace special chars", "remove stopwords", "apply stemmer",
                                     "remove less than two"))

        operations_list = list()
        for op in operations:
            operations_list.append(OPERATIONS[op])
        start = time.time()
        my_bar = st.progress(0)
        with st.spinner('Wait for it...'):
            for operation, name in zip(operations_list, operations):
                df_selected[column] = df_selected[column].apply(operation)
                with st.spinner(f"Applied operation is {name}"):
                    for i in range(100):
                        my_bar.progress(i + 1)
        st.success(f"Processed {len(df_selected)} samples and it's took {(time.time() - start) / 60} minutes.")
        st.write("After operations data frame looks like:")
        st.dataframe(df_selected)
    else:
        st.warning('Please select textual column for operations.')

    elbow = st.slider('Number of clusters for elbow', min_value=2, max_value=100)
    if st.button("Elbow"):
        transformed = apply_tf_idf(df_selected[column])
        df_svd = apply_svd(transformed, 100)
        plot_elbow(df_svd, (1, elbow))
    n_clusters = st.slider('Number of Clusters', min_value=2, max_value=8)
    n_gram = st.slider('N-gram', min_value=1, max_value=4)
    analyzer = st.selectbox("Analyzer for tf-idf.",
                            ["", "word", "char"])

    label = st.checkbox('Label is in data')
    if label:
        label_name = st.selectbox("Select column for label.",
                                  df_selected.columns)
        n_components = st.slider('Number of Components for SVD', min_value=2, max_value=1000)
        if analyzer:
            km, y_, vec_matrix_svd = pipeline(df_selected[column].values, n_cluster=n_clusters,
                                              labels=df_selected[label_name].values, n_components=n_components,
                                              analyzer=analyzer, n_gram=(n_gram, n_gram))
            plot_cluster_result(vec_matrix_svd, y_, km, f"k-means clustering of textual data - k= {n_clusters}")
    else:
        svd = st.checkbox('Use dimension reduction algorithm.')
        if svd:
            transformed = apply_tf_idf(df_selected[column])
            n_components = st.slider('Number of Components for SVD', min_value=2, max_value=1000)
            model = "k-means"
            if analyzer:
                km, y_, vec_matrix_svd = apply_svd_pipeline(df_selected[column], n_clusters=n_clusters,
                                                            n_components=n_components, model=model,
                                                            analyzer=analyzer, n_gram=(n_gram, n_gram))
                plot_cluster_result(vec_matrix_svd, y_, km, f"{model} clustering of textual data - k= {n_clusters}")

        else:
            model = "k-means"
            if analyzer:
                km, y_, vec_matrix_svd = apply_clustering(df_selected[column], n_clusters=n_clusters,
                                                          model=model,
                                                          analyzer=analyzer, n_gram=(n_gram, n_gram))
                plot_cluster_result(apply_svd(vec_matrix_svd, 2), y_, km,
                                    f"{model} clustering of textual data - k= {n_clusters}")
