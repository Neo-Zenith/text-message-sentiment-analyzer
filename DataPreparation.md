# Data Preparation

> We will begin by importing the necessary libraries.
``` Py

import pandas as pd
import numpy as np
import seaborn as sb
import neattext.functions as nfx
from sklearn.model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

```

> We will be creating our model using the **train** dataset. The test and validate datasets are used to check the accuracy of our model.
``` Py

train_data = pd.read_csv("train.txt")

```

