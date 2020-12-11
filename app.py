import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import time
import altair as alt
import PyPDF2
import itertools


st.beta_set_page_config(page_title=None, page_icon="esg_ai_logo.png", layout='centered', initial_sidebar_state='auto')

####### Title ######
col1, col2,col3 = st.beta_columns((1,1,1))
image = Image.open('esg_ai_logo.png')
logo = col2.image(image,width=250)
st.markdown("<hr></hr>",unsafe_allow_html=True)

#splash = st.empty()
#splash.text("Please Load Report Data in sidebar...")


companies = st.multiselect('Select Companies to Analyze',['Apple', 'Microsoft', 'Google', 'Publix'],[])
date_range = st.slider("Select time period",value=[datetime(2020, 1, 1),datetime(2020, 12, 1)],format="MM/DD/YY")







uploaded_file = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True,type=['pdf'])


@st.cache(show_spinner=False,suppress_st_warning=True)
def data_loading(uploaded_file):
	files_info = []

	if uploaded_file is not None and uploaded_file != []:
		splash.text("Processing Documents")
		N = int(100.0/len(uploaded_file))
		my_bar = st.progress(0)
		time.sleep(0.1)
		for idx,i in enumerate(uploaded_file): 
			filename = i.name.replace(".pdf","")
			pdf = PyPDF2.PdfFileReader(i)  
			numPages = pdf.getNumPages()
			text = [pdf.getPage(i).extractText() for i in range(0, numPages)]
			text = "\n".join(text)
			numWords = []
			for sentences in text: 
				numWords.append(len(sentences.split()))
			files_info.append([filename,pdf,numPages,text,sum(numWords)])
			my_bar.progress(int(N*(idx+1)))
		df_files = pd.DataFrame(files_info,columns=['company','pdf_object','num_pages','content','num_words'])
		df_files['estimated_time'] = df_files.num_words.apply(lambda i: str(i/200.0)+ ' mins')
		data_success = st.success('Done!')
		time.sleep(0.5)
		my_bar.empty()
		data_success.empty()
	return df_files

if uploaded_file is not None and uploaded_file != []:
	df_files = data_loading(uploaded_file)
	splash.empty()

	my_expander = st.beta_expander("PDF Report Comparison")
	#my_expander.write('')
	with my_expander:
		col1, col2 = st.beta_columns(2)
		col1.header("Data")
		col1.write(df_files[["company","num_pages","num_words","estimated_time"]])

		col2.header("Visualization")
