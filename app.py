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


#Article Filtering Categories for E, S, and G in the sidebar
esg_categories = st.sidebar.multiselect("Select News Categories",["E","S","G"],["E","S","G"])
print(esg_categories)

####### Title ######
col1, col2,col3 = st.beta_columns((1,1,1))
image = Image.open('esg_ai_logo.png')
logo = col2.image(image,width=250)
st.markdown("<hr></hr>",unsafe_allow_html=True)

###### LOAD DATA ######
@st.cache(show_spinner=False,suppress_st_warning=True)
def load_data(): 
	df_conn = pd.read_csv("Data/connections.csv")
	df_data = pd.read_csv("Data/10daysample.csv",parse_dates=['DATE'],infer_datetime_format=True)
	df_data["DATE"] = df_data["DATE"].dt.date
	companies = df_data.Organization.sort_values(ascending=True).unique().tolist()
	companies.insert(0,"Select a Company")
	return df_conn, df_data, companies
df_conn, df_data,companies= load_data()

#Filter companies by esg category and start and end date
@st.cache(show_spinner=False,suppress_st_warning=True)
def filter_company_data(df_company,esg_categories,start,end):
	print("Filtering...")
	#Filter E,S,G Categories
	comps = []
	for i in esg_categories: 
		X = df_company[df_company[i] == True]
		comps.append(X)
	df_company = pd.concat(comps)
	df_company = df_company[(df_company.DATE >= start) & (df_company.DATE <= end)]
	return df_company

#Filter comapny by publisher once we know what publishers are in the filtered data
@st.cache(show_spinner=False,suppress_st_warning=True)
def filter_publisher(df_company,publisher): 
	if publisher != 'all': 
		df_company = df_company[df_company['SourceCommonName'] == publisher]
	return df_company


###### RUN COMPUTATIONS WHEN A COMPANY IS SELECTED ######
#When a Company is slected show and run the following
company = st.selectbox('Select Companies to Analyze',companies)
date_slider = st.empty()
if company and company != "Select a Company":

	#Filter data to company 
	df_company = df_data[df_data.Organization == company]
	

	#Create date widget
	start = df_company.DATE.min()
	end = df_company.DATE.max()
	selected_dates = st.date_input("Select a Date Range", value=[start,end], min_value=start, max_value=end, key=None)
	time.sleep(0.8)
	start,end = selected_dates
	
	#Filter company data to dates and esg categories
	df_company = filter_company_data(df_company,esg_categories,start,end)

	#Create List of publishers to filter in the sidebar
	publishers = df_company.SourceCommonName.sort_values(ascending=True).unique().tolist()
	publishers.insert(0,"all")
	publisher = st.sidebar.selectbox("Select Publisher",publishers)

	#Filter the company to the selected publisher
	df_company = filter_publisher(df_company,publisher)

	#Create Columns for our company time chart
	st.write("<br>",unsafe_allow_html=True)
	st.write("<center><h3>ESG Trend Over Time</h3></center>",unsafe_allow_html=True)
	st.write("<br>",unsafe_allow_html=True)
	col1,col2 = st.beta_columns((1,3))


	#Create Altair Chart of EST Tone over Time
	line_metric = col1.radio("Choose Metric",options=["Tone","NegativeTone","PositiveTone","Polarity","ActivityDensity","WordCount"])

	df_metr = pd.DataFrame(df_company.groupby(["DATE"])[line_metric].mean().reset_index(),columns=['DATE',line_metric])
	df_metr["DATE"] = pd.to_datetime(df_metr.DATE)
	esg_tone_chart = alt.Chart(df_metr).mark_line(point=True).encode(alt.X('yearmonthdate(DATE):O', title='DATE'),y=line_metric,tooltip=['DATE',line_metric]).interactive()
	col2.altair_chart(esg_tone_chart,use_container_width=True)


	#Expander to Revel Row Level Data
	URL_Expander = st.beta_expander("Data")
	URL_Expander.write(df_company[["DATE","SourceCommonName","URL","Tone","Polarity","ActivityDensity","SelfDensity","WordCount"]])
	st.write("<br><br>",unsafe_allow_html=True)

	#Scatter plot of all articles for the time period
	fake_df = pd.DataFrame(np.random.randn(200, 3),columns=['a', 'b', 'c'])
	#print(df_company.head())
	#df_company["Tone"] = df_company.Tone.astype(float)
	#df_company["WordCount"] = df_company.WordCount.astype(float)
	#print(type(df_company))
	#c = alt.Chart(df_company).mark_circle().encode(x='Tone', y='WordCount', size='c', color='c', tooltip=['Tone', 'WordCount'])
	c = alt.Chart(fake_df).mark_circle().encode(x='a', y='b', size='c', color='c', tooltip=['a', 'b','c'])
	st.altair_chart(c,use_container_width=True)


