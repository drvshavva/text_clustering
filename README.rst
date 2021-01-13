Textual Data Clustering Project
================================

This project provides end-to-end requirements for text clustering.

* Student No: 20501001
* Student : Havvanur Dervisoglu

 * For using run the command below in the project folder:
    $ pip install -e .
 * Documentation directory:
    docs/build/html/index.html
 * For frontend usage run the command below in the frontend folder:
    $ streamlit run frontend.py
 * For command line help run:
    $ nlp --help
 * Example command line:
    $ nlp --input C:\\Users\\user\\Desktop\\nlp_project\\data\\anthems.csv --column_name Anthem --preprocess_operations
     to_lower remove_hyperlink remove_number replace_special_chars remove_punctuation remove_whitespace
     remove_stopwords remove_less_than_two --number_of_cluster 3
    $ nlp --input C:\\Users\\user\\Desktop\\nlp_project\\data\\anthems.csv --column_name Anthem --preprocess_operations
     to_lower remove_hyperlink remove_number replace_special_chars remove_punctuation remove_whitespace remove_stopwords
     remove_less_than_two --number_of_cluster 6 --label Continent --n_components 2 --analyzer char
 * Saved models are in project /models directory
 * Notebooks directory is /notebooks
 * Created images and data quality results are in /results directory
 * Operations directory is /src
        * /src/data/data_quality : To create data quality report of the dataset
        * /src/data/plot : For plot operations.
        * /src/data/rw_utils : Write/Read data operations methods are there.

        * /src/features/feature_engineering : Feature engineering methods.
        * /src/features/preprocess : Preprocess operations methods are here.

        * /src/models/pipelines : Clustering pipelines are in this file.

