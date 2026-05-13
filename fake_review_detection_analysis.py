#!/usr/bin/env python
# coding: utf-8

# ## fake_review_detection_analysis
# 
# 
# 

# # **Data Science Professional Practice Data Science Assessment Project** 
# ## **Fake Review Detection Using Yelp Labeled Review Dataset**
# 
# Fake reviews pose a great threat to consumers and adversely affect fair business practices. As such the identification of fake reviews is increasingly important. This ML project takes a labeled public dataset and adds some features based upon the review metadata and review text.
# A series of ML Classifiers are then tested to determine which gives the best results based upon the evaluation metrics.

# #### **Import Python Modules and Functions**
# Fabric Capacities (the fabric compute) have most of the Common Python libraries installed by default.

# In[1]:


pip install nltk textblob cleantext vadersentiment replaceall


# In[2]:


#general
from datetime import datetime
import numpy as np
import pandas as pd
import re

#plotly express for creating charts in the notebook UI
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

#for text analysis
from replaceall import replaceall
import nltk, cleantext
from nltk.tokenize import word_tokenize
#import scrubadub, spacy, scrubadub_spacy
import vaderSentiment
# calling SentimentIntensityAnalyzer object
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

# StandardScaler
from sklearn.preprocessing import StandardScaler

#machine learning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score, f1_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier


# ### **General Dataframe Functions**
# ##### Dataframe Helper Functions
# 

# In[3]:


def dataframe_row_count(df):
    """
    Return rowcount of pandas dataframe
    """
    return len(df)


# In[4]:


def dataframe_join(df1, df2, join_key_column: list, join_type: str = "inner"):
    """
    Join 2 pandas dataframes based upon common join_key_key column parameter 
    Return joined dataframe
    """
    key1 = join_key_column[0]
    key2 = join_key_column[1]

    # Perform a join and drop column if key1=key2
    if key1 == key2:
        joined_df = df1.merge(df2, on=key1, how=join_type) #.drop(key1)
    else:
        joined_df = df1.merge(df2, df1[key1] == df2[key2], how=join_type)
    
    return joined_df


# ### **Data Cleaning and Preparation**
# ##### Perform data cleaning of dataframes using the dataframe_clean_data() function
# 1. Remove rows with null user_id
# 2. Remove rows with null product_id **(NOTE: 0 is a valid id)**
# 3. Remove rows with null or invalid review_rating (should be an integer range 1-5)
# 4. Remove rows with null or invalid review_date
# 5. Remove rows with empty review_text
# 6. Remove rows with null or invalid label **(NOTE: label is the fake or genuine classification of the review -1=fake, 1= genuine)**
# 

# In[5]:


def dataframe_drop_rows_null_value(df):
    """
    Drops rows from dataframe where the value of any of the dataframe columns is null
    """
    return df.dropna(axis=1,how='any')


# In[6]:


def dataframe_drop_duplicate_rows(df, column_list: list):
    """
    Drops duplicate rows from dataframe using the column list to check for duplicates
    If no columns are given then drop duplicate rows across all columns
    """
    if column_list:
        return df.drop_duplicates(subset=column_list)
    else:
        return df.drop_duplicates() 


# In[7]:


def dataframe_clean_data(df, dataframe_name, duplicated_columns: list):
    """
    Function to perform basic cleaning of dataframe 
    Drops rows where columns in null_columns list are null
    Drops duplicated rows based upon duplicated_column list
    Returns cleaned dataframe
    """
    print(f"Number of rows {dataframe_name} BEFORE removal of null columns = {dataframe_row_count(df)}")

    #drop rows having null value
    df = dataframe_drop_rows_null_value(df)

    print(f"Number of rows {dataframe_name} AFTER removal of null columns = {dataframe_row_count(df)}")

    #drop duplicates
    df = dataframe_drop_duplicate_rows(df, duplicated_columns)

    print(f"Number of rows {dataframe_name} AFTER removal of duplicate columns = {dataframe_row_count(df)}")

    return df


# ### **Load Data for Analysis into Pandas Dataframe**
# ##### The data is then cleaned using the generic cleaning function and the column constraints applied to review_date, review_rating and label

# In[8]:


# Load data into pandas DataFrame from "/lakehouse/default/Files/yelp-labelled-reviews.csv"
df = pd.read_csv("/lakehouse/default/Files/yelp-labelled-reviews.csv")
df.head(10)


# In[9]:


duplicate_column_list  = ["user_id", "product_id","product_rating","review_date","review_text","label"]
df_clean = dataframe_clean_data(df, "yelp_reviews", duplicate_column_list )

#remove rows where review_rating != 1-5 and label != -1,1
df = df_clean.loc[(df_clean["product_rating"].isin([1,2,3,4,5])) & (df_clean["label"].isin([-1,1]))]

#convert review_date to date format
df["review_date"] = pd.to_datetime(df['review_date'])

# Add column containing number of characters in the review_text
df["character_count"] = df["review_text"].apply(len)

# Add column containing number of words in the review_text
df["word_count"] = df["review_text"].apply(lambda x: len(x.split()))

#update label column value -1 to 0
filter = df["label"] == -1
df.loc[filter,"label"] = 0

print(f"Number of rows AFTER removal of out of range columns = {dataframe_row_count(df)}")
print(f"Number of fake reviews = {dataframe_row_count(df.loc[df['label'] == 0])}")
print(f"Number of genuine reviews = {dataframe_row_count(df.loc[df['label'] == 1])}")


# In[10]:


#plot the number of fake and genuine reviews as pie chart
df = df.reset_index()
df_review_count = df.groupby("label", as_index = False ).agg({"index":"count"}).rename(columns = {"index":"count"})
colors = ["Red", "DarkGrey"] 
pie1 = px.pie(df_review_count, values="count", color = "label", names=["Fake", "Genuine"], title="Review Label Distribution",  hole=.2)
pie1.update_traces(textposition="inside", textinfo="value+ percent+label", marker=dict(colors=colors))
pie1.update_layout(width = 500, height = 500, showlegend=True, title ={"x": 0.50,"y": 0.9},legend={"x": 0.3,"y": -0.1, "orientation": "h","yanchor": "top"})
pie1.show()


# In[11]:


#plot the product rating distribution
#1=bad, 2=poor, 3=satisfactory, 4=good 5=excellent.
df_product_rating = df.groupby("product_rating", as_index = False ).agg({"index":"count"}).\
                    rename(columns = {"index":"count"})
colors = [] 
pie2 = px.pie(df_product_rating, values="count", color = "product_rating", names=["1 Bad","2 Poor","3 Satisfactory","4 Good","5 Excellent"],\
                title="Product Rating Distribution",  hole=.2)
pie2.update_traces(textposition='inside', textinfo='value+ percent+label', marker=dict(colors=colors))
pie2.update_layout(width = 500, height = 500, showlegend=True, title ={"x": 0.5,"y": 0.9},\
                    legend={"x": 0.2,"y": -0.1, "orientation": "h","yanchor": "top"})
pie2.show()


# In[12]:


#Number of reviews posted by weekday

# add weekday name and number columns to df
df["weekday"] = df["review_date"].dt.day_name()
df["day_week"] = df["review_date"].apply(lambda x: x.weekday())

df_weekday_dist = df[["index","weekday","day_week","label"]].groupby(["label","weekday","day_week"],\
                    as_index = False).agg({"index":"count"}).rename(columns = {"index":"number"})\
                    .sort_values(by=["label","day_week"])
df_weekday_genuine = df_weekday_dist[df_weekday_dist.label == 1]
df_weekday_fake = df_weekday_dist[df_weekday_dist.label == 0]

bar3 = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], specs=[[{"type": "bar"}, {"type": "bar"}]])

bar3.add_trace(go.Bar(x=df_weekday_genuine.weekday, y=df_weekday_genuine.number, text=df_weekday_genuine.number,\
                name="Genuine", marker_color="DarkGrey"), row=1, col=1)
bar3.add_trace(go.Bar(x=df_weekday_fake.weekday, y=df_weekday_fake.number, text=df_weekday_fake.number, 
                name="Fake", marker_color="Red"), row=1, col=2)
bar3.update_layout(title = "Number of Reviews by Week Day", autosize = False, width = 1000, height = 500, hovermode = False,\
                  showlegend=True, legend={"x": 0.38,"y": -0.3, "orientation": "h","yanchor": "top"})
bar3.update_traces(texttemplate='%{text:.2s}', textposition="inside")
bar3.show()


# ### **Perform Sentiment Analysis of Review Text**
# ##### Perform data cleaning of review text using the cleantext.clean function with cleaning options
# 1. Remove additional spaces, stop words, numbers and punctuation.
# 2. Convert text to lowercase
# 3. Tokenize the text - create tokenized representation of the review_text (typically words, phrases, or sub-words) in order to transform 
# the data into a standard format that machine learning models can process and evaluate
# 

# ##### **Clean review_text column removing spaces, punctuation, stop words, numbers. Perform stemming of the words and convert to lowercase. Optionally tokenize the text**

# In[13]:


def clean_column(dirty_text: str, cleaning_options: dict = {'clean_all':'False','extra_spaces':'True',\
    'stemming':'True','stopwords':'True','stp_lang':'english','lowercase':'True','punct':'True'}, tokenize = True):
    """
    cleantext for general cleaning including regular expressions and stopwords
    tokenization of text is performed by default using nltk word_tokenizer

    clean_all = 'False' # Execute all cleaning operations
    extra_spaces='True' ,  # Remove extra white spaces 
    stemming='True' , # Stem the words
    stopwords='True' ,# Remove stop words
    lowercase='True' ,# Convert to lowercase
    numbers='True' ,# Remove all digits 
    punct='True' ,# Remove all punctuations
    reg: str = '<regex>', # Remove parts of text based on regex
    reg_replace: str = '<replace_value>', # String to replace the regex used in reg
    stp_lang='english'  # Language for stop words
    """
    if dirty_text is not None and len(dirty_text) > 0:
        clean_text = cleantext.clean(dirty_text, cleaning_options)
    if tokenize:
        clean_text = word_tokenize(clean_text)
    return clean_text


# ##### **Apply clean_text function to review text. By default tokenization is performed. Create new column review_text_tokens**

# In[14]:


#apply cleaning and tokenization function to review_text column to create new column review_text_tokens

df["review_text_tokens"] = df["review_text"].apply(clean_column)
df.head(10)


# ## **VADER Sentiment Analyser**
# ##### Create an Instance of the Vader Sentiment Analyser - The VADER (Valence Aware Dictionary and Sentiment Reasoner) is a lexicon and rule-based sentiment analysis tool optimized for social media text excels at identifying sentiment in short, informal texts (slang, emojis, capitalization, punctuation) by providing positive, negative, neutral, and compound scores, with the compound score ranging from -1 (most negative) to +1. 
# ##### **NOTE: Vader uses the raw review_text since case and punctuation are used in the VADER analysis**

# In[15]:


def get_sentiment_polarity(x):
    """
    Return the polarity of the compound sentiment score  1 = Positive, -1 = Negative, 0 = Neutral
    """
    if x>= 0.05:
        return 1
    elif x<= -0.05:
        return -1
    else:
        return 0


# In[16]:


#vader sentimemntIntensityAnalyser
analyser = SentimentIntensityAnalyzer()

df["vader_sentiment_scores"] = df["review_text"].apply(lambda review: analyser.polarity_scores(review))
df["vader_compound_sentiment_score"]  = df["vader_sentiment_scores"].apply(lambda score_dict: score_dict["compound"])
df["vader_sentiment_polarity"] = df["vader_compound_sentiment_score"].apply(get_sentiment_polarity)

df.head(10)


# ## **TextBlob Sentiment Analyser**
# ##### Create an Instance of the TextBlob sentiment analyser which requires the tokenized text
# 

# In[17]:


#textblob sentiment analysis
def textblob_sentiment_polarity(text_tokens):
    """
    Return the polarity of the textblob sentiment analysis score  1 = Positive, -1 = Negative, 0 = Neutral
    """
    text_blob = TextBlob(text_tokens)
    return text_blob.sentiment.polarity


# In[18]:


#change type of tokens list to string
def textblob_token_string(tokens: list):
    """
    Return the tokens list as a string
    """ 
    token_text = replaceall(str(tokens),"[\",\\[\\]]+"," ")
    return token_text


# In[19]:


# Add columns textblob sentiment analysis score and polarity
df["review_text_token_string"] = df["review_text_tokens"].apply(textblob_token_string)
df["textblob_sentiment_score"] = df["review_text_token_string"].apply(textblob_sentiment_polarity)
df["textblob_sentiment_polarity"] = df["textblob_sentiment_score"].apply(get_sentiment_polarity)
df.head(10)


# In[20]:


#plot the distribution of review sentiment polarity for VADER and textblob
#df.drop("level_0")
#df = df.reset_index()

#vader
df_review_count = df.groupby("vader_sentiment_polarity", as_index = False ).agg({"index":"count"}).rename(columns = {"index":"count"})
colors = ["Red", "DarkGrey", "Green"] 
pie4 = px.pie(df_review_count, values="count", color = "vader_sentiment_polarity",\
              names=["Negative", "Neutral", "Positive"], title="VADER Sentiment Analysis Polarity Distribution",  hole=.2)
pie4.update_traces(textposition="inside", textinfo="value+ percent+label", marker=dict(colors=colors))
pie4.update_layout(width = 510, height = 500, showlegend=True, title ={"x": 0.20,"y": 0.9},\
                   legend={"x": 0.2,"y": -0.1, "orientation": "h","yanchor": "top"})
pie4.show()

#textblob
df_review_count = df.groupby("textblob_sentiment_polarity", as_index = False ).agg({"index":"count"}).rename(columns = {"index":"count"})
colors = ["Red", "DarkGrey", "Green"] 
pie5 = px.pie(df_review_count, values="count", color = "textblob_sentiment_polarity",\
              names=["Negative", "Neutral", "Positive"], title="TEXTBLOB Sentiment Analysis Polarity Distribution",  hole=.2)
pie5.update_traces(textposition="inside", textinfo="value+ percent+label", marker=dict(colors=colors))
pie5.update_layout(width = 510, height = 500, showlegend=True, title ={"x": 0.20,"y": 0.9},\
                   legend={"x": 0.2,"y": -0.1, "orientation": "h","yanchor": "top"})
pie5.show()


# ### **Feature Engineering**
# ##### Create new features from existing columns in the dataframe which could improve the performance of Machine Learning Models. The new features will be based upon reviewer statistics and product statistics.
# 
# 1. Average Product Rating **avg_product_rating**
# 2. Total Number of Reviews for each Product **total_product_reviews**
# 3. Maximum Number of Reviews for Product on Review Date **max_product_reviews_date**
# 4. Number of Duplicated Review Texts by Product **count_product_review_text** NOTE: Use the tokenized string representation of the review_text review_text_token_string as malicious reviewers may manually change the text of fake reviews by adding punctuation or numbers to avoid detection
# 5. Average number of words grouped by product **avg_words_product**
# 6. Average Reviewer Rating **avg_user_rating**
# 7. Total Number of Reviews by Reviewer **total_user_reviews**
# 8. Maximum Number of Reviews Submitted by Reviewer on Review Date **max_user_reviews_date**
# 9. Number of Duplicate Review Texts Submitted by Reviewer **count_user_review_text** NOTE: Use the tokenized string representation of the review_text review_text_token_string as malicious reviewers may manually change the text of fake reviews by adding punctuation or numbers to avoid detection
# 10. Average number of words grouped by user **avg_words_user**
# 

# In[22]:


#1.Average Product Rating avg_product_rating
df["avg_product_rating"] = df.groupby("product_id")["product_rating"].transform("mean")

#2.Total Number of Reviews for each Product total_product_reviews
df["total_product_reviews"] = df.groupby("product_id")["product_id"].transform("count")

#3.Maximum Number of Reviews for Product on Review Date max_product_reviews_date
df_maxcount_review_product = df.groupby(["product_id","review_date"], as_index = False).agg({"review_text":"count"})\
                        .drop("review_date",axis = 1).groupby(["product_id"],as_index = False).agg({"review_text":"max"})\
                        .rename(columns = {"review_text":"max_product_reviews_date"})
#join the dataframes on product_id
df = dataframe_join(df, df_maxcount_review_product, join_key_column = ["product_id","product_id"], join_type = "inner")

#4.Number of Duplicated Review Texts by Product count_distinct_review_text_product
#create dataframe of count distinct review_text_token_string grouped by product_id
df_distinctcount_review_product = df.groupby("product_id")["review_text_token_string"].nunique().reset_index()\
                                .rename(columns = {"review_text_token_string":"countdistinct_review_token_product"})
#join the dataframes on product_id
df = dataframe_join(df, df_distinctcount_review_product, join_key_column = ["product_id","product_id"], join_type = "inner")

#5. Average number of words grouped by product
df["avg_words_product"] = df.groupby("product_id")["word_count"].transform("mean")

#6.Average Reviewer Rating avg_user_rating
df["avg_user_rating"] = df.groupby("user_id")["product_rating"].transform("mean")

#7.Total Number of Reviews by Reviewer total_user_reviews
df["total_user_reviews"] = df.groupby("user_id")["user_id"].transform("count")

#8.Maximum Number of Reviews Submitted by Reviewer on Review Date max_user_reviews_date
df_maxcount_review_user = df.groupby(["user_id","review_date"], as_index = False).agg({"review_text":"count"})\
                        .drop("review_date",axis = 1).groupby(["user_id"],as_index = False).agg({"review_text":"max"})\
                        .rename(columns = {"review_text":"max_user_reviews_date"})
#join the dataframes on product_id
df = dataframe_join(df, df_maxcount_review_user, join_key_column = ["user_id","user_id"], join_type = "inner")

#9.Number of Duplicate Review Texts Submitted by Reviewer
#create dataframe of count distinct review_text_token_string grouped by user_id
df_distinctcount_review_user = df.groupby("user_id")["review_text_token_string"].nunique().reset_index()\
                                .rename(columns = {"review_text_token_string":"countdistinct_review_token_user"})
#join the dataframes on user_id
df = dataframe_join(df, df_distinctcount_review_user, join_key_column = ["user_id","user_id"], join_type = "inner")

#10. Average number of words grouped by user
df["avg_words_product"] = df.groupby("user_id")["word_count"].transform("mean")

df.head(10)


# ### **Machine Learning Analysis**
# ##### The Machine Learning Analysis will train a variety of Logistic Regression Classification models on a subset of the data then test the model on the remaining data. The following models have been selected for comparison and assessment of ther relative performance using the evaluation metrics.
# 
# 

# #### **Select Columns from DataFrame to be used as Features in ML Analysis**
# **Columns:** "day_week","product_rating","label","word_count","character_count","textblob_sentiment_score",\
#         "textblob_sentiment_polarity","avg_product_rating","count_distinct_review_text_token_string"\
#         "max_product_reviews_date","countdistinct_review_token_product","avg_words_product","avg_user_rating",\
#         "total_user_reviews","max_user_reviews_date","countdistinct_review_token_user","avg_words_user"

# In[23]:


#create analysis dataframe analysis dataframe. Select columns to be used as features

df_analysis  = df.filter(["day_week","product_rating","label","word_count","character_count","textblob_sentiment_score",\
                        "textblob_sentiment_polarity","avg_product_rating","count_distinct_review_text_token_string"\
                        "max_product_reviews_date","countdistinct_review_token_product","avg_words_product","avg_user_rating",\
                        "total_user_reviews","max_user_reviews_date","countdistinct_review_token_user","avg_words_user"],  axis=1)

#fill nulls with zero
df_analysis.fillna(value=0, inplace=True)
df_analysis.head(10)


# #### **Create Dictionary of Machine Learning Models to Train and Test**
# ##### A comparison of the models will be performed using the model evaluation metrics.  

# In[24]:


# Machine Learning Classifiers
KN = KNeighborsClassifier()
DT = DecisionTreeClassifier(max_depth=5)
LR = LogisticRegression(solver='liblinear', penalty='l1')
RF = RandomForestClassifier(n_estimators=50, random_state=2)
AB = AdaBoostClassifier(n_estimators=50, random_state=2)
XGB = XGBClassifier(n_estimators=50,random_state=2)

#build dictionary of ml models 
ml_classifier_models = {
    "knear-neigh" : KN, 
    "desc-tree": DT, 
    "log-reg": LR, 
    "rand-forest": RF, 
    "ada-boost": AB, 
    "xgb-boost": XGB 
}


# #### **Functions to Train and Test the ML Model and Evaluate the Metrics**
# ##### dataframe_evaluation_metrics also plots the confusion matrix

# In[25]:


# Normalised confusion matrix
def display_confusion_matrix(y_test, pred, classes, model_name, normalize=False,title=None,cmap=plt.cm.OrRd):
    """
    Calculate the confusion matrix from the test dataframe and predictions
    """
    if not title:
        if normalize:
            title = "Normalized Confusion Matrix " + str(model_name)
        else:
            title = "Absolute Confusion Matrix " + str(model_name)

    # Compute confusion matrix
    c_matrix = confusion_matrix(y_test, pred)

    if normalize:
        c_matrix = c_matrix.astype("float") / c_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Absolute Confusion Matrix")

    plt.style.use("default")

    fig, ax = plt.subplots(figsize = (6,6), dpi = 70)
    im = ax.imshow(c_matrix, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(c_matrix.shape[1]),
           yticks=np.arange(c_matrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel="True label",
           xlabel="Predicted label")

    # Rotate labels and set alignment.
    plt.title(title, y = 1.06)
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = c_matrix.max() / 2.
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            ax.text(j, i, format(c_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="black")
    fig.tight_layout()
    return ax


# #### **Split Data into Training and Test Data Frames**
# ##### The training test split will be 80:20 percent.  

# In[26]:


df_training, df_testing = train_test_split(df_analysis, test_size = 0.2, random_state = 666)


# #### **Implement Under Sampling Techniques as the dataset is imbalanced with significantly more genuine than fake reviews**
# #####  Random Under-sampling randomly deletes rows with the predominant label to achieve a more balanced training dataset. The python code implements random under sampling.

# In[27]:


df_train_0 = df_training[df_training.label == 0]
df_train_1 = df_training[df_training.label == 1]
df_train_0_sample = df_train_0.sample(n = df_train_1.shape[0], random_state=42, replace=True) 
df_train_1_sample = df_train_1.sample(n = df_train_1.shape[0], random_state=42, replace=True) 
df_train_sample = pd.concat([df_train_0_sample, df_train_1_sample])

print("Number of fake reviews in Training Dataset: " + str(df_train_sample[df_train_sample.label == 0].shape))
print("Number of genuine reviews in Training Dataset: " + str(df_train_sample[df_train_sample.label == 1].shape))

#create label (df_y) and features (df_X) datasets from random under sampled training data
df_y_train_rus = df_train_sample["label"]
df_X_train_rus = df_train_sample.drop(["label"], axis = 1)

print("Number of fake reviews in Label Training Dataset after Random Under Sampling: "\
    + str(df_y_train_rus.shape))
print("Number of genuine reviews in label Training Dataset after Random Under Sampling: "\
    + str(df_y_train_rus.shape))
    
#create label (df_y) and features (df_X) datasets from test data
df_y_test = df_testing["label"]
df_X_test = df_testing.drop(["label"], axis = 1)


# #### **Implement Normalization or Scaling of the Features Dataframe**
# #####  This technique ensures that features with different scales contribute equally to machine learning models which is essential for distance-based algorithms like K-Nearest Neighbors and Support Vector Machines.

# In[28]:


# StandardScaler
standard_scaler = StandardScaler()
#apply to features(df_X) dataframes
df_X_train_scaled = standard_scaler.fit_transform(df_X_train_rus)
df_X_test_scaled = standard_scaler.transform(df_X_test)


# #### **Iterate through Each ML Model - Train and Fit the Model to Test Data**
# ##### The Evaluation metrics are added to an array for comparison and the Confusion Matrix is displayed using function

# In[29]:


accuracy_scores = []
precision_scores = []
f1_scores = []
recall_scores = []
auroc_scores = []

#iterate through dict of ML Models
for name,ml_model in ml_classifier_models.items(): 
       
    ml_model.fit(df_X_train_scaled,df_y_train_rus)
    y_pred = ml_model.predict(df_X_test_scaled)
    accuracy = accuracy_score(df_y_test,y_pred)
    precision = precision_score(df_y_test, y_pred, average='weighted')
    f1 = f1_score(df_y_test, y_pred, average='weighted')
    recall = recall_score(df_y_test, y_pred, average='weighted')
    auroc = roc_auc_score(df_y_test, y_pred, average='weighted')
    
    #display confusion matrix as heatmap
    display_confusion_matrix(df_y_test, y_pred, classes=["0","1"], model_name=name, normalize=True)
    
    # add metrics to arrays
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    f1_scores.append(f1)
    recall_scores.append(recall)
    auroc_scores.append(auroc)


# In[31]:


#create a dataframe from the evaluation metric arrays for each ML model

performance_df = pd.DataFrame({"Machine Learning Classifier":ml_classifier_models.keys(),"Accuracy":accuracy_scores,\
"Precision":precision_scores,"F1":f1_scores,"Recall": recall_scores, "AUROC": auroc_scores}).\
sort_values("Precision",ascending=False)

display(performance_df)


# In[ ]:




