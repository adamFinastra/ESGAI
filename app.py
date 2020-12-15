import streamlit as st
# import numpy as np
import pandas as pd
# from datetime import datetime
# from PIL import Image
# import time
import altair as alt
# import PyPDF2
# import itertools



def main():
    ####### Title ######
    col1, col2,col3 = st.beta_columns((1,1,1))
    # image = Image.open('esg_ai_logo.png')
    # logo = col2.image(image,width=250)
    col2.image("esg_ai_logo.png", width=250)
    st.markdown("<hr></hr>",unsafe_allow_html=True)

    
    ###### LOAD DATA ######
    df_conn, df_data,companies= load_data()



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
    	start,end = selected_dates
    	
    	#Filter company data to dates 
    	print(start,end)
    	df_company = df_company[(df_company.DATE >= start) & (df_company.DATE <= end)]
    	print(df_company.shape)

    	#Create ESG Tone over time 
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


@st.cache(show_spinner=False, suppress_st_warning=True,
          allow_output_mutation=True)
def load_data(): 
    df_conn = pd.read_csv("Data/connections.csv")
    df_data = pd.read_csv("Data/dec1dec2sample.csv",parse_dates=['DATE'],infer_datetime_format=True)
    df_data["DATE"] = df_data["DATE"].dt.date
    companies = df_data.Organization.unique().tolist()
    companies.insert(0,"Select a Company")
    return df_conn, df_data, companies
    
    
    
    
if __name__ == "__main__":
    st.set_page_config(page_title="ESG AI", page_icon="esg_ai_logo.png",
                       layout="centered", initial_sidebar_state="auto")

    main()






