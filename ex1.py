import pandas as pd
import numpy as np
df=pd.read_csv('language.csv')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
x=np.array(df['Text'])
y=np.array(df['language'])
vc=CountVectorizer()
x=vc.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
model=MultinomialNB()
model.fit(x_train,y_train)
import pickle
with open('language1.pkl', 'wb') as f:
    pickle.dump(model, f)

import streamlit as st

st.title('Language Detection App')
user = st.text_input('Enter some text to detect the language:')
if st.button('Detect Language'):
    # Transform the user's input text into a numerical representation
    df = vc.transform([user]).toarray()
    # Use the trained model to predict the language
    outputs = model.predict(df)
    print(outputs)
    st.write(f'Detected Language: {outputs}')