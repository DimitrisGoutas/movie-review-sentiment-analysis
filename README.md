# Sentiment Analysis of Movie Reviews

A comparison of three sentiment analysis approaches on 50,000 movie reviews:

* **VADER**, a lexicon and rule based method
* **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment`), a transformer trained on tweets
* **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`), a transformer fine-tuned on SST-2

None of the models were trained on movie reviews. The goal is to see how well they work out of the box and to understand where and why they fail.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| VADER | 0.697 | 0.650 | 0.858 | 0.739 |
| RoBERTa | 0.861 | 0.894 | 0.820 | 0.855 |
| DistilBERT | 0.889 | 0.916 | 0.856 | 0.885 |

Main findings from the error analysis:

* VADER has a strong positive bias. It is easily misled by positive words that do not reflect the overall opinion, for example when a reviewer praises the early seasons of a show and then criticises what it has become. Over 10,000 reviews were misclassified only by VADER.
* RoBERTa leans towards negative predictions and gives a high neutral score to the reviews it gets wrong, which points to a domain mismatch between tweets and long movie reviews.
* DistilBERT is almost as confident in its wrong predictions as in its correct ones.
* All models struggle with reviews that describe a dark or violent plot while expressing a positive opinion indirectly.
* Accuracy drops for long reviews, and much more for the transformers, which only read the first 512 tokens.

## Dataset

The notebook uses the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle. The dataset is not included in this repository. Download it and save it as `mvreviews.csv` in the same folder as the notebook.

## How to run

```
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

The RoBERTa and DistilBERT parts run on a GPU (`cuda`). Running all three models on the full dataset takes a while.

## Project structure

```
analysis.ipynb     the full analysis
requirements.txt   Python packages used
```
