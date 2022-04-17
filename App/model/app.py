# import the necessary packages
import streamlit as st
import altair as alt
from sklearn.calibration import CalibratedClassifierCV
import plotly.express as px 

#EDA packages
import pandas as pd
import numpy as np
from datetime import datetime

import joblib
import emoji

pipe_lsv = joblib.load(open("emotion_classifier_pipe_lsv.pkl","rb"))

# function for emotion prediction
def predict_emotion(docx):
    results = pipe_lsv.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lsv.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger":u"\U0001F620","fear":u"\U0001F628","joy":u"\U0001F604","love":u"\U0001F970","sadness":u"\U0001F622","surprise":u"\U0001F929"}

# functions for the data record
import sqlite3
conn = sqlite3.connect('record_data.db')
c = conn.cursor()

def create_page_visited_table():
	c.execute('CREATE TABLE IF NOT EXISTS pageTrackTable(pagename TEXT,timeOfvisit TIMESTAMP)')

def add_page_visited_details(pagename, timeOfvisit):
    c.execute('INSERT INTO pageTrackTable(pagename,timeOfvisit) VALUES(?,?)', (pagename,timeOfvisit))
    conn.commit()

def create_page_visited_table():
	c.execute('CREATE TABLE IF NOT EXISTS pageTrackTable(pagename TEXT,timeOfvisit TIMESTAMP)')

def view_all_page_visited_details():
	c.execute('SELECT * FROM pageTrackTable')
	data = c.fetchall()
	return data

def create_emotionclf_table():
	c.execute('CREATE TABLE IF NOT EXISTS emotionclfTable(rawtext TEXT,prediction TEXT,probability NUMBER,timeOfvisit TIMESTAMP)')

def add_prediction_details(rawtext,prediction,probability,timeOfvisit):
	c.execute('INSERT INTO emotionclfTable(rawtext,prediction,probability,timeOfvisit) VALUES(?,?,?,?)',(rawtext,prediction,probability,timeOfvisit))
	conn.commit()

def view_all_prediction_details():
	c.execute('SELECT * FROM emotionclfTable')
	data = c.fetchall()
	return data


# main fucntion
def main():
    st.title("Text-based Emotion Classifier App")
    menu = ["Home","Record","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    create_page_visited_table()
    create_emotionclf_table()

    if choice == "Home":
        add_page_visited_details("Home",datetime.now())
        st.subheader("Home - Emotion In Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2 = st.columns(2)

            # Apply the functions here
            prediction = predict_emotion(raw_text)
            probability = get_prediction_proba(raw_text)

            add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{} : {}".format(prediction,emoji_icon))
                st.write("Confidence : {}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")   
                # st.write(probability)
                proba_df = pd.DataFrame(probability,columns=pipe_lsv.classes_)
                proba_df.rename(columns={0:'Probability'},inplace=True)
                st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                st.altair_chart(fig,use_container_width=True)

    elif choice == "Record":
        add_page_visited_details("Record",datetime.now())
        st.subheader("History record")

        with st.beta_expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
            st.altair_chart(pc,use_container_width=True)	

    else:
        st.subheader("About")
        add_page_visited_details("About",datetime.now())


if __name__ == '__main__':
    main()
    

