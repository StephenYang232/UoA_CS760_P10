# UoA_CS760_P10
Recommender System based on Sentiment analysis


## Envernment

.ipynb files can run on Google Colab directly. And .R file needs R 4.3 or higher version.

** You don't need to buy any plan from Colab Pro. Free is enough.

## Data

Our data is from https://www.kaggle.com/datasets/venkatasubramanian/%73%65%6e%74%69%6d%65%6e%74-%62%61%73%65%64-%70%72%6f%64%75%63%74-%72%65%63%6f%6d%6d%65%6e%64%61%74%69%6f%6e-%73%79%73%74%65%6d/. You can re-save to your google drive from easier using.

## Running

If you use the code on the Colab.
Check the code Loading data part. Make sure your code can read your data correctly. And press the button on the left. That is running. Easy!

Checking the following things please:
### Roberta

the pre-processed files are named train.csv and test.csv, run Roberta.ipynb, and a result dataset result.csv will be automatically generated.

#### XLNet

You don't need to change the name of the data, and the result will also show in your directory

## Outcome

It will be shown following the code.

### Roberta outcome

The dataset will store a column of the id and a column of the prediction results Y_pred_roberta, as well as the other two datasets trainclean.csv and testclean.csv, which are used to store the cleaned data.

### XLNet

You can see the result will be showen in the directory and other intermediate results will be there just as well
