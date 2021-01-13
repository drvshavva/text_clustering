from src.models.pipelines import apply_svd_pipeline, apply_clustering, pipeline
from src.features.preprocess import remove_punctuation, remove_hyperlink, remove_less_than_two, remove_number, \
    remove_stopwords, remove_whitespace, replace_special_chars, apply_stemmer, to_lower
import warnings
import sys
import os
import argparse
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

PREPROCESS_OPS = {"to_lower": to_lower, "remove_hyperlink": remove_hyperlink, "remove_number": remove_number,
                  "remove_punctuation": remove_punctuation,
                  "remove_whitespace": remove_whitespace,
                  "replace_special_chars": replace_special_chars,
                  "remove_stopwords": remove_stopwords, "apply_stemmer": apply_stemmer,
                  "remove_less_than_two": remove_less_than_two}


def plot_cluster_result(data, predictions: list, model, title: str):
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
    return plt


def parameter_parser(args):
    if args.input is not None:
        df = pd.read_csv(args.input)
        df = df.dropna()
        df = df.drop_duplicates()

        if args.preprocess_operations is not None:
            operations = list()
            for operation in args.preprocess_operations:
                operations.append(PREPROCESS_OPS[operation])

            columns = df.columns
            if args.column_name is not None or args.column_name not in columns:
                start = time.time()
                for operation in operations:
                    df[args.column_name] = df[args.column_name].apply(operation)
                print(f"Processed {len(df)} samples.\n")
                print(f"It's took {(time.time() - start) / 60} minutes.")
            else:
                print(f"Selected column must be in data frame columns. Columns: {columns}")
                exit(0)
            if args.use_svd is True and args.label is None:
                clustering, y_predictions, data_svd = apply_svd_pipeline(data=df[args.column_name],
                                                                         n_components=args.n_components,
                                                                         n_clusters=args.number_of_cluster,
                                                                         model=args.clustering,
                                                                         idf=args.use_idf,
                                                                         analyzer=args.analyzer,
                                                                         n_gram=(
                                                                             args.ngram_range[0], args.ngram_range[1]))
            elif args.label is not None:
                clustering, y_predictions, data_svd = pipeline(corpus=df[args.column_name].values,
                                                               labels=df[args.label].values,
                                                               n_components=args.n_components,
                                                               n_cluster=args.number_of_cluster,
                                                               idf=args.use_idf,
                                                               analyzer=args.analyzer,
                                                               n_gram=(
                                                                   args.ngram_range[0], args.ngram_range[1]))
            else:
                clustering, y_predictions, data_svd = apply_clustering(data=df[args.column_name],
                                                                       n_clusters=args.number_of_cluster,
                                                                       model=args.clustering,
                                                                       idf=args.use_idf,
                                                                       analyzer=args.analyzer,
                                                                       n_gram=(
                                                                           args.ngram_range[0], args.ngram_range[1]))
            plot_cluster_result(data_svd, y_predictions, clustering,
                                f"{args.clustering} clustering of news data").show()

        else:
            print("Preprocess operations must be selected.")
            exit(0)
    else:
        print("Input file path must be given.")
        exit(0)


def main():
    parser = argparse.ArgumentParser(prog='Text Clustering Library',
                                     description='Text Clustering Library,'
                                                 'create machine learning clustering pipeline from command line',
                                     usage='%(prog)s [OPTIONS]')
    parser.add_argument("--input",
                        type=str,
                        dest="input",
                        default=None,
                        help="Input file path with extension, for example: /usr/bin/data.csv ")
    parser.add_argument("--column_name",
                        type=str,
                        dest="column_name",
                        default=None,
                        help="Select one of the column for use preprocess operations and clustering.")
    parser.add_argument("--label",
                        type=str,
                        dest="label",
                        default=None,
                        help="If label column is in data name of it.")
    parser.add_argument("--preprocess_operations",
                        dest="preprocess_operations",
                        default=None,
                        type=str,
                        nargs='+',
                        choices=["to_lower", "remove_hyperlink", "remove_number", "remove_punctuation",
                                 "remove_whitespace",
                                 "replace_special_chars", "remove_stopwords", "apply_stemmer", "remove_less_than_two"],
                        help="Select preprocess operations")
    parser.add_argument("--use_idf",
                        type=bool,
                        dest="use_idf",
                        default=True,
                        help="Use idf or not")
    parser.add_argument("--analyzer",
                        type=str,
                        dest="analyzer",
                        choices=["word", "char"],
                        default="word",
                        help="Whether the feature should be made of word or character n-grams.")
    parser.add_argument("--ngram_range",
                        type=int,
                        nargs=2,
                        dest="ngram_range",
                        default=[1, 1],
                        help="The lower and upper boundary of the range of n-values for different n-grams to be"
                             " extracted. All values of n such that min_n <= n <= max_n will be used.")
    parser.add_argument("--use_svd",
                        type=bool,
                        dest="use_svd",
                        default=True,
                        help="Use svd dimension reduction or not")
    parser.add_argument("--n_components",
                        type=int,
                        default=20,
                        dest="n_components",
                        help="Number of components for singular value decomposition operation.")
    parser.add_argument("--clustering",
                        type=str,
                        choices=['k-means', 'k-medoids', 'dbscan', 'ahct', 'mean-shift'],
                        default="k-means",
                        dest="clustering",
                        help="Select one of the clustering method to apply. If label is not None no need to select.")
    parser.add_argument("--number_of_cluster",
                        type=int,
                        dest="number_of_cluster",
                        help="Number of cluster or component for clustering operation.")

    args = parser.parse_args(sys.argv[1:])
    sys.stdout.write(str(parameter_parser(args)))


if __name__ == "__main__":
    sys.exit(main())
