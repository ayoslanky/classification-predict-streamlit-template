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
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Data dependencies
import pandas as pd
import numpy as np

# Vectorizer
news_vectorizer = open("resources/tranformer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv") 
def extract_handles(x):
    """ This function finds handles in a 
        tweet and returns them as a list"""
    handles = []
    for i in x:
        h = re.findall(r'@(\w+)', i)
        handles.append(h)
        
    return handles

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	# Creates a main title and subheader on your page -
	# these are static across all pages

  

    
    
	col_1, col_2 = st.columns(2)
	with col_1:
		st.title("IAH DATA SOLUTIONS")
	with col_2:
		st.image('https://my-03321245515-bucket.s3.amazonaws.com/Screenshot+(282).png')
	
	
    
 	# Creates a main title and subheader on your page -
	# these are static across all pages

	with st.sidebar:
          selected = option_menu("Main Menu", ['Home','About','Information', 'Prediction'],icons=['house', 'emoji-smile','info-square-fill','graph-up-arrow'], menu_icon="cast", default_index=0)
          selected
          
              

    
		


    
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	
	selection = selected

	# Building out the "Home" page
	if selection == "Home":
		st.info("Home")
  
		# You can read a markdown file from supporting resources folder
	
		col1, col2,  = st.columns(2)
		with col1:
			st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://images.pexels.com/photos/590022/pexels-photo-590022.jpeg?cs=srgb&dl=pexels-lukas-590022.jpg&fm=jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
			st.write('Our Amazing products and Services Stands out')
   
		with col2:
			st.image("https://images.pexels.com/photos/5561912/pexels-photo-5561912.jpeg?cs=srgb&dl=pexels-olya-kobruseva-5561912.jpg&fm=jpg")
   
 	
	if selection == "About":
		st.info("We believe every business should be empowered to make data driven decisions")
		# You can read a markdown file from supporting resources folder
		

		col3, col4,  = st.columns(2)
		with col3:
			st.image("https://images.pexels.com/photos/7567211/pexels-photo-7567211.jpeg?cs=srgb&dl=pexels-tima-miroshnichenko-7567211.jpg&fm=jpg")
   
		with col4:
			st.image("https://images.pexels.com/photos/4960464/pexels-photo-4960464.jpeg?cs=srgb&dl=pexels-george-morina-4960464.jpg&fm=jpg")
        
      	
	# Building out the "Information" page
	if selection == "Information":
		st.info("Current Project")
		# You can read a markdown file from supporting resources folder
		st.image("https://images.pexels.com/photos/60013/desert-drought-dehydrated-clay-soil-60013.jpeg?cs=srgb&dl=pexels-pixabay-60013.jpg&fm=jpg")
			
		st.subheader("Climate Change Twitter belief Analysis")
		st.markdown("The project aims at building a robust machine learning model which will empower the marketing team of a business to inform  their strategies. By providing insight and analysis of the sentiments of potential consumers generated from their tweet data, the business will be able  increase sales exponentially and save cost")

		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']].head(5)) # will write the df to the page

		if st.checkbox('Extract handles'): # data is hidden if box is unchecked
			news_h = extract_handles(raw['message']
                              [raw['sentiment'] == 2])
			pro_h = extract_handles(raw['message']
                          [raw['sentiment'] == 1])
			neutral_h = extract_handles(raw['message']
                              [raw['sentiment'] == 0])
			anti_h = extract_handles(raw['message']
                          [raw['sentiment'] == -1])
			handles = [sum(news_h, []), sum(pro_h, []), sum(neutral_h, []),
           sum(anti_h, [])]
			full_title = ['Impact of Handles on the News sentiment',
              'Impact of Handles on the Pro sentiment',
              'Impact of Handles on the Neutral sentiment',
              'Impact of Handles on the Anti sentiment']
			plt.rcParams['figure.figsize'] = [50, 5]
			fig = plt.figure(figsize = (10, 5))

			for i, sent in enumerate(handles):
				#plt.subplot(1)
				freq_dist = nltk.FreqDist(sent)
				df = pd.DataFrame({'Handle': list(freq_dist.keys()),
                      'Count' : list(freq_dist.values())})
				df = df.nlargest(columns='Count', n=10)
				ax = sns.barplot(data=df, y='Handle', x='Count', palette="tab20c")
				plt.title(full_title[i])
				st.pyplot(fig)


       
   
    

    
    
		
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
            # When model has successfully run, will print prediction
			st.success("{}".format(sentiment[prediction[0]]))
			
     
     
       
			
		uploaded_file = st.file_uploader("Choose a file")
		if uploaded_file is not None:
		# Can be used wherever a "file-like" object is accepted:
			dataframe = pd.read_csv(uploaded_file)  
			st.write(dataframe)
    	
    		
		if st.button("Predict on your data"):
			
			vect_frame = tweet_cv.transform(dataframe['message'])
			predictor = joblib.load(open(os.path.join("resources/ridge_model.pkl"),"rb"))
			dataframe['sentiment'] = predictor.predict(vect_frame)
			st.write(dataframe)
			


			plt.rcParams['figure.figsize'] = [50, 5]
			fig2 = plt.figure(figsize = (10, 5))


        
			class_dist = pd.DataFrame(list(dataframe['sentiment'].value_counts()),index=['Pro', 'News', 'Neutral', 'Anti'],columns=['Count'])
			# Plot class distribution	
			sns.set(style="whitegrid")
			sns.barplot(x=class_dist.index, y=class_dist.Count,palette="Set3")
			plt.title('Class Distributions') 
			st.pyplot(fig2)




           
    
            


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
