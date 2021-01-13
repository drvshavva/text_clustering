import matplotlib.pyplot as plt
from pandas import DataFrame
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import warnings
from sklearn.cluster import KMeans
import seaborn as sns
import scipy.cluster.hierarchy as sch
from os.path import dirname
import matplotlib.cm as cm

warnings.filterwarnings(action='ignore')
plt.style.use('ggplot')

RESULTS_PATH = dirname(dirname(dirname(__file__))) + "/results/images/"


def plot_wordcloud(corpus: list, max_words: int = 200, title: str = None) -> WordCloud:
    """
    This method generates wordcloud for given corpus

    :param corpus: list of sentences, :type str
    :param max_words: maximum word count, :type int
    :param title: label for sentences, :type str
    :return: WordCloud
    """
    comment_words = ''
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                          stopwords=stopwords,
                          max_words=max_words,
                          max_font_size=100,
                          random_state=42,
                          width=800,
                          height=400,
                          mask=None)

    for sent in corpus:
        comment_words += " " + sent

    wordcloud.generate(str(comment_words))

    plt.figure(figsize=(24.0, 16.0))
    plt.imshow(wordcloud);
    plt.title(title, fontdict={'size': 40, 'color': 'black',
                               'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}word_cloud_{title}")
    return plt


def plot_bar_chart(labels: list, values: list, title: str):
    """
    This method plot bar chart

    :param labels: list of labels, :type list
    :param values: count of each label values, :type list
    :param title: title of plot
    :return: plot
    """
    y_pos = np.arange(len(labels))
    plt.figure(figsize=(24.0, 16.0))
    plt.bar(y_pos, values, align='center')
    plt.xticks(y_pos, labels)
    plt.ylabel('Count')
    plt.title(title)
    plt.savefig(f"{RESULTS_PATH}bar_chart_{title}")
    return plt


def plot_pie_chart(labels: list, values: list, title: str):
    """
    This method plot pie chart

    :param labels: list of labels, :type list
    :param values: count of each label values, :type list
    :param title: title of plot
    :return: plot
    """
    plt.figure(figsize=(24.0, 16.0))
    plt.pie(values, labels=labels, startangle=90, autopct='%.1f%%')
    plt.title(title)
    plt.savefig(f"{RESULTS_PATH}pie_chart_{title}")
    return plt


def plot_count_plot(label_name: str, data: DataFrame, title: str):
    """
    This method returns count plot of the dataset

    :param label_name: name of the class, :type str
    :param data: input dataFrame, :type DataFrame
    :param title: title of plot
    :return plt
    """
    plt.figure(figsize=(24.0, 16.0))
    sns.countplot(x=label_name, data=data)
    plt.title(title)
    plt.savefig(f"{RESULTS_PATH}plot_count_{title}")
    return plt


def plot_dendogram(data: DataFrame):
    """
    This method plots dendogram to decide number of clusters in dataset

    :param data: train dataset, :type DataFrame
    :return dendogram
    """
    plt.figure(figsize=(24.0, 16.0))
    dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('data')
    plt.ylabel('Euclidean distances')
    plt.savefig(f"{RESULTS_PATH}dendogram")
    return plt


def plot_elbow(data: DataFrame, num_of_cluster_range: set):
    """
    This method provides elbow ploting result

    :param num_of_cluster_range: range of the clusters, :type set
    :param data: input data frame, :type DataFrame,
    :return plot
    """
    wcss = list()
    for i in range(num_of_cluster_range[0], num_of_cluster_range[1]):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(24.0, 16.0))
    plt.plot(range(num_of_cluster_range[0], num_of_cluster_range[1]), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('wcss')
    plt.savefig(f"{RESULTS_PATH}elbow_method")
    return plt


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
    plt.figure(figsize=(24.0, 16.0))
    for i in range(0, number_of_class):
        plt.scatter(data[predictions == i, 0], data[predictions == i, 1], s=100, c=colors[i], label=f'cluster{str(i)}')
    try:
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300, c='blue', label='Centroids')
    except:
        print("used model has no cluster centers")
    plt.title(title)
    plt.legend()
    plt.savefig(f"{RESULTS_PATH}{title}")
    return plt
