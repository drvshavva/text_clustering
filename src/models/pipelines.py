from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from pandas import DataFrame
from sklearn_extra.cluster import KMedoids
import pickle
from os.path import dirname
import collections
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
MODELS_PATH = dirname(dirname(dirname(__file__))) + "/models/"


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
    label = km.labels_

    resulted_data = pd.DataFrame({'Completeness': [metrics.completeness_score(labels, km.labels_)],
                                  'Homogeneity': [metrics.homogeneity_score(labels, km.labels_)],
                                  'silhouette_score ': [
                                      metrics.silhouette_score(vec_matrix_svd, label, metric='euclidean')]})
    print(resulted_data)

    with open(MODELS_PATH + pkl_filename, 'wb') as file:
        pickle.dump(km, file)

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


def apply_pca(data: DataFrame, n_components: int) -> DataFrame:
    """
    This method applies pca to given dataset

    :param data: input dataframe, :type DataFrame
    :param n_components: number of result dataframe features,, :type int
    :return DataFrame
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


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
        print("Wrong clustering type selected !!")
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


def apply_pca_pipeline(data: DataFrame, n_components: int, model: str, max_df: int = 1, min_df: int = 1,
                       idf: bool = True, max_features: int = None, analyzer: str = "word", n_gram=(1, 1), **kwargs):
    """
    This method applies pca dimension reduction algorithm to dataset and then clusters data with using selected clustering
    algorithm

    :param n_gram:
    :param min_df:
    :param idf:
    :param analyzer:
    :param max_features:
    :param max_df:
    :param data: DataFrame
    :param n_components: number of result dataframe features,, :type int
    :param model: selected clustering algorithm, :type str
         ['k-means', 'k-medoids', 'dbscan', 'ahct', 'mean-shift']
    :return model
    """
    pkl_filename = f"pca_{model}_{min_df}_{max_features}_{n_components}_{n_gram}.pkl"

    transformed = apply_tf_idf(data, max_df, min_df, idf, max_features, analyzer, n_gram)
    data_pca = apply_pca(transformed, n_components)
    print(f"Shape of the data:{data_pca.shape}")

    cluster = get_model(model, data, **kwargs)
    if cluster is None:
        print("Wrong clustering type selected !!")
        return
    y_ = cluster.fit_predict(data_pca)

    print(f"Number of samples in each cluster:{collections.Counter(y_)}")

    with open(MODELS_PATH + pkl_filename, 'wb') as file:
        pickle.dump(cluster, file)

    return cluster, y_, data_pca


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
    print(f"Shape of the data:{data_svd.shape}")

    cluster = get_model(model, data, **kwargs)
    if cluster is None:
        print("Wrong clustering type selected !!")
        return
    y_ = cluster.fit_predict(data_svd)
    label = cluster.labels_

    resulted_data = pd.DataFrame({
                                  'silhouette_score ': [
                                      metrics.silhouette_score(data_svd, label, metric='euclidean')]})
    print(resulted_data)
    print(f"Number of samples in each cluster:{collections.Counter(y_)}")

    with open(MODELS_PATH + pkl_filename, 'wb') as file:
        pickle.dump(cluster, file)

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
    print(f"Shape of the data:{data.shape}")

    transformed = apply_tf_idf(data, max_df, min_df, idf, max_features, analyzer, n_gram)
    cluster = get_model(model, transformed, **kwargs)
    if cluster is None:
        print("Wrong clustering type selected !!")
        return
    y_ = cluster.fit_predict(transformed)
    label = cluster.labels_

    resulted_data = pd.DataFrame({
        'silhouette_score ': [
            metrics.silhouette_score(data_svd, label, metric='euclidean')]})
    print(resulted_data)

    print(f"Number of samples in each cluster:{collections.Counter(y_)}")

    with open(MODELS_PATH + pkl_filename, 'wb') as file:
        pickle.dump(cluster, file)

    return cluster, y_, apply_svd(transformed, 2)


def get_resulted_df(data: list, labels: list):
    """
    This method concatenates data and labels and returns the data frame

    :param data: data of the data frame
    :param labels: labels
    :return:
    """
    list_of_tuples = list(zip(data, labels))

    df_labeled = pd.DataFrame(list_of_tuples,
                              columns=['data', 'labels'])
    return df_labeled
