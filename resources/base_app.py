"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies

import streamlit as st
import joblib,os
from streamlit_option_menu import option_menu

# Data dependencies
import pandas as pd
import numpy as np

# Vectorizer
news_vectorizer = open("resources/tranformer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	# Creates a main title and subheader on your page -
	# these are static across all pages

	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification") 
 	# Creates a main title and subheader on your page -
	# these are static across all pages

	with st.sidebar:
          selected = option_menu("Main Menu", ["Home",'about','Information', 'Prediction'],icons=['house', 'emoji-smile','info-square-fill','graph-up-arrow'], menu_icon="cast", default_index=1)
          selected
          
              

    



    
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	
	selection = selected#st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
        
  
		if st.button("Stochastic Gradient Descent"):
			st.header("Running SGDC model")
			vect_text = tweet_cv.transform([tweet_text])
			predictor = joblib.load(open(os.path.join("resources/sgd_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			sentiment={1:'You believe in man-made climate change', 2:'The tweet links to factual news about climate change', 0:'The tweet neither supports nor refutes the belief of man-made climate change', -1:'The tweet does not believe in man-made climate change'}   
			st.success("{}".format(sentiment[prediction[0]]))  
   
    
		if st.button("Ridge Classifier"):
			st.header("Running the ridge classifier model")
			vect_text = tweet_cv.transform([tweet_text])
			predictor = joblib.load(open(os.path.join("resources/ridge_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			sentiment={1:'You believe in man-made climate change', 2:'The tweet links to factual news about climate change', 0:'The tweet neither supports nor refutes the belief of man-made climate change', -1:'The tweet does not believe in man-made climate change'}   
			st.success("{}".format(sentiment[prediction[0]]))  
   
		if st.button("Support Vector classifier"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text])
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/svc_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			sentiment={1:'You believe in man-made climate change', 2:'The tweet links to factual news about climate change', 0:'The tweet neither supports nor refutes the belief of man-made climate change', -1:'The tweet does not believe in man-made climate change'}   
			st.success("{}".format(sentiment[prediction[0]]))
			
     
     
        
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
            
            
			st.success("{}".format(sentiment[prediction[0]]))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
