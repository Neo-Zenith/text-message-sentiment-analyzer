# Text-based Emotion Classifier

## Preface
### Objective(s)
>  In this highly-digitalised day and age, texting has become the preferred way of communication for the current generation. However, texting has indirectly impacted the art of communicating - through the negligence of emotion. Consequently, text messages can often be misinterpreted, depending on the perspectives of the perceiver and sender. <br>
>  
> The main objective of this project is to utilise the knowledge we learnt in elementary data science and machine learning to build a simple application bsaed on the following key factors:
> * To predict the emotion of a text message at a reasonable accuracy.
> * To provide the predicted probability of each emotion from the given sentence to account for cases where a multitude of emotions are present.
>
> In addition, we figured a few potential routes to take our simple application further into development in the future:
> * Sentimental analysis of customer review on online products.
> * Sentimental analysis of IMDB ratings of movies.
> * Online dating profile matching algorithm fine-tuning based on the general perception of emotion from a conversation.

### Skills Learnt
> * Perform Exploratory Data Analysis on unstructured data (texts) using Word Cloud.
> * Concepts about Recall, Precision & F1-score.
> * Logistic Regression, Linear Support Vector Machine & Naive Bayes Algorithm implementation in Machine Learning.
> * Implementation of Cross-Validation Check.
> * Implementation of an application's graphical user interface using Streamlit.
> * Elementary Object-Oriented Programming during the standardization of functions & classes.
> * Introduction to documentation writing.
> * Collaboration using GitHub.

### Dataset 
#### Source of Dataset
> https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp by Praveen

#### Format of Dataset
> | text         | emotion |
> |--------------|---------|
> |i didnt feel humiliated | sadness |
> |i can go from feeling so hopeless to so damned hopeful just from being around... | sadness |
> |im grabbing a minute to post i feel greedy wrong | anger |
> |i am ever feeling nostalgic about the fireplace i will know that it is still... | love |
> 
> **Note:** ***text*** and ***emotion*** are separated by a semi-colon ***';'***.
<pre>
i didnt feel humiliated;sadness
i am feeling grouchy;anger
...
</pre>

### Contributors 
> This mini-project is co-authored by:
> * Lee Juin (Alias: [Neo-Zenith](https://github.com/Neo-Zenith))
> * Kassim bin Mohamad Malaysia (Alias: [kassimmalaysia](https://github.com/kassimmalaysia))
> * Lee Ci Hui (Alias: [perfectsquare123](https://github.com/perfectsquare123))

### Default Libraries
> The following libraries are used throughout the project. 
> * [Pandas](https://pandas.pydata.org/docs/) <br>
> * [Numpy](https://numpy.org/doc/stable/) <br>
> * [Seaborn](https://seaborn.pydata.org/tutorial.html) <br>
> * [NLTK](https://www.nltk.org) <br>
> * [Word Cloud](https://www.lfd.uci.edu/~gohlke/pythonlibs/#wordcloud) <br>
> * [Matplotlib](https://matplotlib.org/3.5.1/)<br>
> * [Scikit-Learn](https://scikit-learn.org/stable/)
> 
> **Note:** `Word Cloud` has not received any official support for Python 3.8x and above. Thus, we used [Word Cloud *unofficial*](https://www.lfd.uci.edu/~gohlke/pythonlibs/#wordcloud) as our library instead. For Python 3.7x and below, please refer to [Word Cloud](https://pypi.org/project/wordcloud/). However, do note that our project is ran and tested on Python 3.8x and above.

### Custom Libraries
> We have compiled a list of functions and classes which are useful during our project. These functions are repeatedly used within our project, and can be found in [Libraries](https://github.com/Neo-Zenith/SC1015-GP/blob/main/Libraries.py). <br>
> 
> Please read [Libaries Information](https://github.com/Neo-Zenith/SC1015-GP/blob/main/Libraries%20Information.md) for the details of the functions and classes found within our custom library.

## Miscellaneous 
### Updates
> For the progress of our project, please refer to [Updates](https://github.com/Neo-Zenith/SC1015-GP/blob/main/Updates.md).

### Issues
#### [FIXED] Issue on Jupyter Notebook (Ipynb files) and Github  <br>
> There appears to be a widespread issue ongoing on Github w.r.t the incorrect printing/inability to print outputs from Jupyter Notebook formatted files. <br> <br>
> **Replicable:** `Yes` <br>
> **Source of Issue:** Most likely `Github` <br>
> **Fixed:** `Yes` <br>
> **Comments:** Please use alternative IDE to inspect the main code sections. [Visual Studio Code](https://code.visualstudio.com) is known to be working properly. 

## Run-through
### Overview
> Our code section is divided into `3` main portion: <br>
> * [Data Preparation](https://github.com/Neo-Zenith/SC1015-GP/edit/main/README.md#data-preparation)
> * [Exploratory Data Analysis](https://github.com/Neo-Zenith/SC1015-GP/edit/main/README.md#exploratory-data-analysis)
> * [Machine Learning](https://github.com/Neo-Zenith/SC1015-GP/edit/main/README.md#machine-learning)

### Data Preparation
> In this section, we perform the necessary import of libraries, as well as our train dataset. We also performed simple analysis of our dataset to get a brief outlook of what kind of data we were dealing with. <br>
> 
> Please refer to [Text-based Emotion Classifier](https://github.com/Neo-Zenith/SC1015-GP/blob/main/Text-based%20Emotion%20Classifier.Ipynb) for the details of our source code.

### Exploratory Data Analysis
> In this section, we perform mainly more in-depth analysis of our dataset. From the analysis, we figured out that our dataset requires some cleaning. Thus, we have performed dataset cleaning which can be classified into the following 3 phases:
> * Lemmatization of words
> * Removal of HTML tags and attributes
> * Removal of stopwords
> 
> We are mainly using the [NLTK](https://www.nltk.org) library as our de-facto dataset cleaning library. <br>
> 
> We are mainly using the [Word Cloud](https://www.lfd.uci.edu/~gohlke/pythonlibs/#wordcloud) as our main data visualisation library. <br>
> 
> Please refer to [Text-based Emotion Classifier](https://github.com/Neo-Zenith/SC1015-GP/blob/main/Text-based%20Emotion%20Classifier.Ipynb) under `Exploratory Data Analysis` for the details of our source code.

### Machine Learning
> In this section, we perform machine learning by using the following 3 models on our train dataset:
> * Logisitc Regression
> * Naive Bayes Algorithm
> * Linear Support Vector Machine
> 
> We proceeded to apply our trained models on the validation dataset, and obtain their respective **Precision**, **Recall** and **F1-socre**. <br>
> 
> We further performed a **repeated k-fold cross validation check** on each model to determine the best model from the three. <br>
> 
> Finally we apply the best model we chose on the test dataset.
> 
> Please refer to [Text-based Emotion Classifier](https://github.com/Neo-Zenith/SC1015-GP/blob/main/Text-based%20Emotion%20Classifier.Ipynb) under `Machine Learning` for the details of our source code.