# UoA_CS760_P10
Recommender System based on Sentiment analysis


## Envernment

.ipynb files can run on Google Colab directly.
.R file needs R 4.3 or higher version. Use Install.packages("dplyr").

** You don't need to buy any plan from Colab Pro. Free is enough.

## Data

Our data is from https://www.kaggle.com/datasets/venkatasubramanian/%73%65%6e%74%69%6d%65%6e%74-%62%61%73%65%64-%70%72%6f%64%75%63%74-%72%65%63%6f%6d%6d%65%6e%64%61%74%69%6f%6e-%73%79%73%74%65%6d/. You can re-save to your google drive from easier using.

** If you want to use the R code, the Python code can give you the data needed by the R code.

## Running

If you use the code on the Colab.
Check the code Loading data part. Make sure your code can read your data correctly. And press the button on the left. That is running. Easy!

### Roberta

The original data file with csv format

### XLNet

You don't need to change the name of the data, and the result will also show in your directory

### .R file

Open that with R or RStudio. Just run it!

** Remember the data can be found in the Python code results.

## Outcome

It will be shown following the code.

### Roberta

A result dataset result.csv, which will store one column of ids and one column of predictions Y_pred_roberta, and four other datasets train.csv and test.csv, trainclean.csv, and testclean.csv, the first two datasets store the data that are unprocessed and are only used for dividing the training and test sets, respectively. divide the data between training and test sets, and the last two datasets are used to store the cleaned training and test set data respectively



### XLNet

You can see the result will be showen in the directory and other intermediate results will be there just as well
