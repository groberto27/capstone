import pandas as pd
import numpy as np
import streamlit as st
from docx import Document
import pdfplumber
import csv #Already within Base Python
import io
import os
import json
from textwrap import dedent
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")
import openai
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity 
import spacy
nlp = spacy.load("en_core_web_lg")
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


#####################################################################

csv_file_path = '/Users/nicholasreese/Desktop/Georgetown/Capstone/capstone_github/Capstone_Introduction/Saxa-4-Capstone/spacy_redacted_documents.csv'
data = pd.read_csv(csv_file_path)
json_file_path = '/Users/nicholasreese/Desktop/Georgetown/Capstone/capstone_github/Capstone_Introduction/Saxa-4-Capstone/redacted_resumes_output.json'

data.to_json(json_file_path, orient = 'records', lines = True)

resumes = data

# This below is used for reading in anytype of resume.
st.markdown('# Recommendation System')
st.markdown('---')
#####################################################################

st.write('## Section 1 - Loading Resume')
st.markdown('### Upload your Resume')

uploaded_file = st.file_uploader('Drag & and Drop your resume. We can analyze word, pdf, txt or csv',
                                 type = ['docx', 'txt', 'pdf', 'csv'])


# Word
if uploaded_file is not None:
    resume_df = None
    def read_docx(file):
        doc = Document(file)
        data = [para.text for para in doc.paragraphs]
        return pd.DataFrame(data, columns = ['Redacted Text'])
# Text
    def read_txt(file):
        data = file.read().decode("utf-8").splitlines()
        return pd.DataFrame(data, columns = ['Redacted Text'])
# Pdf
    def read_pdf(file):
        with pdfplumber.open(file) as pdf:
            data = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    data.append(text)
            return pd.DataFrame({
                'Redacted Text': [file]
                })
# CSV

    def read_csv(file):
        df = pd.read_csv(file)
        return df
    
    if uploaded_file.name.endswith('.docx'):
        resume_df = read_docx(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        resume_df = read_txt(uploaded_file)
    elif uploaded_file.name.endswith('.pdf'):
        resume_df = read_pdf(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        resume_df = read_csv(uploaded_file)
    else:
        st.error("Unsupported file type!")


st.markdown('### Non Processed Resume')
st.dataframe(resume_df)

#######################################################################

st.markdown('---')
st.markdown('## Section 2 - LLM')

st.markdown('### Recommender Output')
st.markdown('#### this initial iteration will use a default prompt')

os.environ['OPENAI_API_KEY'] = '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'  # Replace with your actual key
client = OpenAI(api_key=os.environ.get("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")) 
#Verify that the environment variable is set
print(os.environ.get('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'))

#######################################################################
# Adding Stopwords and Vectorizing 

sw = stopwords.words("english")
sw.extend(['[Redacted]'])

vec = TfidfVectorizer(stop_words = sw)

################################

# Calculating Distance 


def calc_sim(resume_df, resumes):
    
    new_resume = str(resume_df)
    
    resumes['Redacted Text'] = resumes['Redacted Text'].astype(str)
    
    all_resumes = resumes['Redacted Text'].tolist() + [new_resume]
    
    vec = TfidfVectorizer(stop_words = sw)
    tfidf_matrix = vec.fit_transform(all_resumes)
    
    similarity_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    similar_indices = similarity_matrix.argsort()[0][-5:][::-1]
    
    return similar_indices

##################################################
# Adding in the function to get recs

resume_rec_prompt = '''
You are a helpful reccommender. You will be provided a resume where the individual would like to have a different
job reccommended to them. Your goal will be to provide five similar roles to what is on their resume that they provided.
For each reccommendation, please provide an explanation as to why you chose the job you did for the individual.
'''
new_resume = resume_df

def get_rec_roles(new_resume, resumes):
    
    #new_resume = resume_df
    similar_indices = calc_sim(new_resume, resumes)
    
    resumes_texts = resumes['Redacted Text'].tolist()
    
    selected_resumes = [resumes_texts[i] for i in similar_indices]
    
    prompt = resume_rec_prompt + '\n\n' + '\n'.join(selected_resumes)
    
    try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature= 0.5
    )
        return response.choices[0].message.content
    except Exception as e:
        print(f' an error occurred at {e}')
        return None

result = get_rec_roles(new_resume, resumes)

st.markdown(result)

################################

## Adding the Default Prompt
default_prompt = '''
You are a helpful recommender. You will be provided a resume where the individual would like to have a different
job recommended to them. Your goal will be to provide five similar jobs to what is on their resume that they provided.
For each recommendation, please provide an explanation as to why you chose the job you did for the individual.
'''
#####################################

def calc_sim(resume_df, resumes):
    
    new_resume = str(resume_df)
    
    resumes['Redacted Text'] = resumes['Redacted Text'].astype(str)
    
    all_resumes = resumes['Redacted Text'].tolist() + [new_resume]
    
    vec = TfidfVectorizer(stop_words = sw)
    tfidf_matrix = vec.fit_transform(all_resumes)
    
    similarity_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    similar_indices = similarity_matrix.argsort()[0][-5:][::-1]
    
    return similar_indices



def get_rec_roles2(new_resume, resumes, user_prompt):
    new_resume = resume_df
    similar_indices = calc_sim(new_resume, resumes)
    resumes_texts = resumes['Redacted Text'].tolist()
    selected_resumes = [resumes_texts[i] for i in similar_indices]

    prompt = user_prompt + '\n\n' + '\n'.join(selected_resumes)
    try:
        response = client.chat.completions.create(
            model = 'gpt-3.5-turbo',
            messages= [
                {'role': 'user',
                 'content': prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f'An error occured: {e}')
        return None

st.markdown('---')


def main():
    st.markdown("### Choose your own prompt")

    user_prompt = st.text_area('Below you can ask your own prompt. If you have a specific industry in mind, you can see what skills might help you acquire a role', 
                               value= default_prompt)
    
    if st.button('Get Recommendations'):
        if new_resume is not None:
            final_prompt = user_prompt.strip() if user_prompt.strip() else default_prompt
            
            recommendations = get_rec_roles2(new_resume, resumes, final_prompt)
            if recommendations:
                st.write('### Recommendations:')
                st.write(recommendations)
            else:
                st.write('No Recs were generated')
        else: 
            st.write('Please enter your resume')
if __name__ == '__main__':
    main()    

st.markdown('---')
#######################################################################

st.markdown('## Section 3 - PII')

st.write("""
         In issue that the team wanted to tackle with their Capstone was the redaction of Personally Identifiable Information or PII. There are many reasons why the team may want to redact PII,
         but some that were the most prevelent were bias, privacy and accountability. 

         The team wanted to test whether names of inidividuals, names of schools, names of former employeers, locations and other identifiable information produced a different output for a resume that had 
         PII redacted and a resume without PII redacted. 

         The same can be said about privacy and accountability, the team wanted to highlight the importance of keeping your information private. 
         """)

st.markdown('---')

