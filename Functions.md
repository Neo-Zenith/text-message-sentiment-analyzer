### Data Visualisation using WordCloud

> The function requires the [WordCloud](https://www.lfd.uci.edu/~gohlke/pythonlibs/#wordcloud) library
``` Py

# Import the required library
from wordcloud import WordCloud, STOPWORDS

# Main function
# data is any pandas DataFrame
# column is a string variable that corresponds to the name of the column intended to be used for visualisation
# bg is a string variable that takes in input for the colour of the background ('black', 'white')
# max_words is an integer variable that dictates the maximum number of words allowed to be displayed on the visuals
# max_font_size is an integer variable that dictates the maximum font size of a single word in the visuals
# scale is an integer variable that dictates the scale of the visual, lower value makes the visual more blurry, but larger
# figsize is a tuple of integers that dictates the size of the visual
def show_wordcloud(data, column, bg, max_words, max_font_size, scale, figsize):
  
    # Setting the stopwords
    # Creating a string of all the words, separated by commas
    stopwords = set(STOPWORDS)
    text = "".join(t for t in data[column])
    
    # Sub-function to display the visuals
    def display():
        wordcloud = WordCloud(
            background_color = bg,
            stopwords = stopwords,
            max_words = max_words,
            max_font_size = max_font_size,
            scale = scale,
            random_state = 1)
      
        wordcloud = wordcloud.generate(str(text))

        fig = plt.figure(1, figsize = figsize)
        plt.axis('off')

        plt.imshow(wordcloud)
        plt.show()

    display()

```

> An example of using the above function:
``` Py

show_wordcloud(cleaned_train, 'clean_text', 'black', 500, 30, 3, (20, 20))

```

> The result is as follows: <br>
> ![alt text](https://github.com/Neo-Zenith/SC1015-GP/blob/main/raw/Visual.png)
