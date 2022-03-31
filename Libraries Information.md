### NLP Model Class

> We have created a class which contains the methods for building the model, fitting the model and predicting the accuracy of our model.

``` Py

class NLPModel:
    def __init__(self, dataset, model, model_name, x, y):
        self.dataset = dataset
        self.model = model
        self.name = model_name
        self.x = x
        self.y = y

    # Return a tuple that contains the columns of x (text) and columns of y (emotion) in the dataset
    def getLabels(self):
        xLabel = self.dataset[self.x]
        yLabel = self.dataset[self.y]
        return (xLabel, yLabel)
    
    # Build the pipeline of the model specified.
    def buildPipeline(self):
        pipeline = Pipeline(steps = [('cv', CountVectorizer()), (self.name, self.model)])
        return pipeline
    
    # Feed the train data into the pipeline to train the model
    def fitModel(self, pipeline, xLabel, yLabel):
        pipeline.fit(xLabel, yLabel)
        return pipeline
    
    # Return a tuple that contains a list of prediction based on the dataset, and the classification accuracy of the model
    def score(self, pipeline, xLabel, yLabel):
        prediction = pipeline.predict(xLabel)
        predScore = pipeline.score(xLabel, yLabel)
        return (prediction, predScore)
    
    # Return the Confusion Matrix of the model
    def buildCM(self, pred, actual):
        CM = confusion_matrix(actual, pred)
        return CM
    
    # Print out the Confusion Matrix
    def printCM(self, CM):
        plt.figure(figsize = (20, 20))
        sb.heatmap(CM, annot = True, fmt=".0f", annot_kws={"size": 18})
    
    # Return a dataframe which contains the Recall, Precision and F1-score of the model
    def printScore(self, predScore, CM):
        print("The classification accuracy is: ", predScore)
        print("")

        precision = []
        recall = []
        f1 = []

        row = 0
        col = 0
        total = 0
        for i in pd.Categorical(self.dataset[self.y]).categories:
            for j in range(len(pd.Categorical(self.dataset[self.y]).categories)):
                total += CM[row][col]
                row += 1

            precision.append(CM[col][col] / total)
            col += 1
            row = 0
            total = 0

        row = 0
        col = 0
        total = 0
        for i in pd.Categorical(self.dataset[self.y]).categories:
            for j in range(len(pd.Categorical(self.dataset[self.y]).categories)):
                total += CM[row][col]
                col += 1

            recall.append(CM[row][row] / total)
            row += 1
            col = 0
            total = 0

        index = 0
        for i in pd.Categorical(self.dataset[self.y]).categories:
            f1score = (2 * precision[index] * recall[index]) / (precision[index] + recall[index])
            f1.append(f1score)
            index += 1
        
        score_df = pd.DataFrame()
        score_df['Emotion'] = pd.Categorical(self.dataset[self.y]).categories
        score_df['Precision'] = precision
        score_df['Recall'] = recall
        score_df['F1-score'] = f1

        return score_df

```

### WordCloud Display

> We will be using WordCloud as our primary means of visualising our data.

``` Py

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
    
```

### Data Cleaning Functions

> Below are the relevant functions we created for cleaning our dataset.

``` Py

# Function which lemmatizes words.
# Lemmatize words means => 'bringing' will be lemmatized to 'bring'
# Lemmatization is done by inspecting the Part-of-Speech before deciding if lemmatization should occur.
# 'feeling' may be classified as adjective, i.e. will not be lemmatized into 'feel' in certain context.
def lemmatizeWords(dataset, column, newColumn):
    lm = WordNetLemmatizer()
    lemmatized_data = []

    for sentence in dataset[column]:
        sentence = "".join([lm.lemmatize(w, 'v') for w in sentence])
        lemmatized_data.append(sentence)

    dataset[newColumn] = lemmatized_data 

    return dataset

# Obtain the html keywords from online source
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

# Remove the html keywords present in a text
def remove_html_attr(text, html_keys):
    text = text.split()
    return " ".join([t for t in text if t.lower() not in set(html_keys)])

# Extend words which are missing their apostrophes
def extend_word(word):
    custom_words = set(['dont', 'didnt', 'shouldnt', 'cant', 'wont', 'wouldnt', 'musnt'])

    temp = []
    if word in custom_words:
        for char in range(0, len(word) - 1):
            temp.append(word[char])
   
        return "".join(temp)
    
    return word

# Main function that removes stopwords from the texts.
# Stopwords library is based on NLTK's stopwords.
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

```
