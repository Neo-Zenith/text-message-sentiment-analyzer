from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud, STOPWORDS

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

class NLPModel:
    def __init__(self, dataset, model, model_name, x, y):
        self.dataset = dataset
        self.model = model
        self.name = model_name
        self.x = x
        self.y = y

    def getLabels(self):
        xLabel = self.dataset[self.x]
        yLabel = self.dataset[self.y]
        return (xLabel, yLabel)
    
    def buildPipeline(self):
        pipeline = Pipeline(steps = [('cv', CountVectorizer()), (self.name, self.model)])
        return pipeline

    def fitModel(self, pipeline, xLabel, yLabel):
        pipeline.fit(xLabel, yLabel)
        return pipeline
    
    def score(self, pipeline, xLabel, yLabel):
        prediction = pipeline.predict(xLabel)
        predScore = pipeline.score(xLabel, yLabel)
        return (prediction, predScore)

    def buildCM(self, pred, actual):
        CM = confusion_matrix(actual, pred)
        return CM
    
    def printCM(self, CM):
        plt.figure(figsize = (20, 20))
        sb.heatmap(CM, annot = True, fmt=".0f", annot_kws={"size": 18})

    def printScore(self, predScore, CM):
        print("The classification accuracy is: ", predScore)
        print("")

        T_Anger_Rate = CM[0][0] / (CM[0][0] + CM[1][0] + CM[2][0] + CM[3][0] + CM[4][0] + CM[5][0])
        F_Anger_Rate = 1 - T_Anger_Rate

        T_Fear_Rate = CM[1][1] / (CM[0][1] + CM[1][1] + CM[2][1] + CM[3][1] + CM[4][1] + CM[5][1])
        F_Fear_Rate = 1 - T_Fear_Rate

        T_Joy_Rate = CM[2][2] / (CM[0][2] + CM[1][2] + CM[2][2] + CM[3][2] + CM[4][2] + CM[5][2])
        F_Joy_Rate = 1 - T_Joy_Rate

        T_Love_Rate = CM[3][3] / (CM[0][3] + CM[1][3] + CM[2][3] + CM[3][3] + CM[4][3] + CM[5][3])
        F_Love_Rate = 1 - T_Love_Rate

        T_Sadness_Rate = CM[4][4] / (CM[0][4] + CM[1][4] + CM[2][4] + CM[3][4] + CM[4][4] + CM[5][4])
        F_Sadness_Rate = 1 - T_Sadness_Rate

        T_Surprise_Rate = CM[5][5] / (CM[0][5] + CM[1][5] + CM[2][5] + CM[3][5] + CM[4][5] + CM[5][5])
        F_Surprise_Rate = 1 - T_Surprise_Rate

        print("True Anger Rate:\t", T_Anger_Rate)
        print("False Anger Rate:\t", F_Anger_Rate)
        print("")
        print("True Fear Rate:\t\t", T_Fear_Rate)
        print("False Fear Rate:\t", F_Fear_Rate)
        print("")
        print("True Joy Rate:\t\t", T_Joy_Rate)
        print("False Joy Rate:\t\t", F_Joy_Rate)
        print("")
        print("True Love Rate:\t\t", T_Love_Rate)
        print("False Love Rate:\t", F_Love_Rate)
        print("")
        print("True Sadness Rate:\t", T_Sadness_Rate)
        print("False Sadness Rate:\t", F_Sadness_Rate)
        print("")
        print("True Surprise Rate:\t", T_Surprise_Rate)
        print("False Surprise Rate:\t", F_Surprise_Rate)


def show_wordcloud(data, column, bg, max_words, max_font_size, scale, figsize):
    stopwords = set(STOPWORDS)
    text = " ".join(t for t in data[column])
    def display():
        wordcloud = WordCloud(
            background_color = bg,
            stopwords = stopwords,
            max_words = max_words,
            max_font_size = max_font_size,
            scale = scale,
            random_state = 1)
    
        wordcloud = wordcloud.generate(str(text))

        plt.figure(1, figsize = figsize)
        plt.axis('off')

        plt.imshow(wordcloud)
        plt.show()

    display()


def lemmatizeWords(dataset, column, newColumn):
    lm = WordNetLemmatizer()
    lemmatized_data = []

    for sentence in dataset[column]:
        sentence = "".join([lm.lemmatize(w, 'v') for w in sentence])
        lemmatized_data.append(sentence)

    dataset[newColumn] = lemmatized_data 

    return dataset

def get_html_keys():
    url = "https://www.w3schools.com/TAgs/default.asp"
    html_keys = pd.read_html(url)
    html_tags = []
    html_keys[0].head()

    for i in html_keys[0]['Tag']:
        i = i.strip("<>")
        html_tags.append(i)
    
    html_tags.append("www")
    html_tags.append("http")
    html_tags.append("https")

    url = "https://www.w3schools.com/tags/ref_attributes.asp"
    html_keys = pd.read_html(url)
    html_attr = []
    html_keys[0].head()

    for i in html_keys[0]['Attribute']:
        html_attr.append(i)

    return html_tags + html_attr


def remove_html_attr(text, html_keys):
    text = text.split()
    return " ".join([t for t in text if t.lower() not in set(html_keys)])


def extend_word(word):
    custom_words = set(['dont', 'didnt', 'shouldnt', 'cant', 'wont', 'wouldnt', 'musnt'])

    temp = []
    if word in custom_words:
        for char in range(0, len(word) - 1):
            temp.append(word[char])
   
        return "".join(temp)
    
    return word

def removeStopWords(dataset, column):
    # Custom stopwords which are not in the nltk stopwords set
    custom_stopwords = set(['im', 'ive', 'ill', 'feeling', 'feel', 'felt'])

    cleaned_data = []
    outlier = []
    forbidden_words = set(stopwords.words('English'))
    loc = 0

    for sentence in dataset[column]:

        # Extend words that are concatenated without apostrophe such as 'cant' to 'can' and 't'
        # Remove the individual words which are present in the custom stopwords set and the nltk stopwords set
        split_sentence = sentence.split()
        temp = []
        for word in split_sentence:
            word = extend_word(word)
            if word not in custom_stopwords and word not in forbidden_words:
                temp.append(word)
        
        # Join back the words which are not removed into a sentence
        sentence = " ".join(temp)
        
        # Treat a sentence with only stopwords and html tags and attributes as outliers
        if len(sentence) == 0:
            outlier.append(loc)

        cleaned_data.append(sentence)
        loc += 1
    
    return (outlier, cleaned_data)