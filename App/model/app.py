import streamlit as st

#EDA pkgs
import pandas as pd
import numpy as np

#Utils
import joblib

def main():
    st.title("Emotion Classifier App")
    menu = ["Home","Monitor","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home-Emotion In Text")

    elif choice == "Monitor":
        st.subheader("Monitor App")

    else:
        st.subheader("About")

    