# Data Preparation

> We will begin by importing the necessary libraries.
``` Py

import pandas as pd
import numpy as np
import seaborn as sb
import neattext.functions as nfx
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

```

> We will be creating our model using the **train** dataset. The test and validate datasets are used to check the accuracy of our model.
``` Py

import os
dir = os.getcwd()
os.chdir("..")

train_data = pd.read_csv("datasets/train.txt", sep = ';')
os.chdir(dir)

```

> We start by cleaning our dataset.
``` Py

train_data['clean_text'] = train_data['text'].apply(nfx.remove_stopwords)
train_data['clean_text'] = train_data['clean_text'].apply(nfx.remove_userhandles)

```

> Finally, we save our cleaned dataset into a new file to be accessed by subsequent processes.
``` Py

dir = os.getcwd()
os.chdir("..")

train_data.to_csv('datasets/clean_train.csv', sep = ',', header = True)
os.chdir(dir)

```
