import streamlit as st
# import numpy as np
import pandas as pd
from datetime import datetime
import time
import altair as alt
import pickle
import itertools
import networkx as nx
import nx_altair as nxa
from Graph import graph_creator
import plotly.express as px


st.set_page_config(page_title="ESG AI", page_icon="esg_ai_logo_1.png",
                   layout='centered', initial_sidebar_state='auto')
st.image("esg_ai_logo.png", use_container_width=True)
###### LOAD DATA ######
@st.cache(show_spinner=False, suppress_st_warning=True,
          allow_output_mutation=True)
def load_data():
    df_conn = pd.read_csv("Data/newGconnections_10days.csv")
    df_data = pd.read_csv("Data/10daysample.csv",parse_dates=['DATE'],infer_datetime_format=True)
    df_data["DATE"] = df_data["DATE"].dt.date
    companies = df_data.Organization.sort_values(ascending=True).unique().tolist()
    companies.insert(0,"Select a Company")
    embeddings = pd.read_csv("Data/newGembeddings_10day.csv")
    return df_conn, df_data, companies, embeddings

df_conn, df_data,companies, embeddings = load_data()

####### Title ######
col1, col2, col3 = st.beta_columns(3)
# st.image("esg_ai_logo.png", width=200)
# # st.markdown("<hr></hr>", unsafe_allow_html=True)

####### Sidebar ######
esg_categories = st.sidebar.multiselect("Select News Categories",
                                        ["E", "S", "G"], ["E", "S", "G"])


#Filter companies by esg category and start and end date
@st.cache(show_spinner=False, suppress_st_warning=True)
def filter_company_data(df_company, esg_categories, start, end):
    #Filter E,S,G Categories
    comps = []
    for i in esg_categories:
        X = df_company[df_company[i] == True]
        comps.append(X)
    df_company = pd.concat(comps)
    df_company = df_company[(df_company.DATE >= start) &
                            (df_company.DATE <= end)]
    return df_company


#Filter comapny by publisher once we know what publishers are in the filtered data
@st.cache(show_spinner=False,suppress_st_warning=True)
def filter_publisher(df_company,publisher):
    if publisher != 'all':
        df_company = df_company[df_company['SourceCommonName'] == publisher]
    return df_company


#Create ESG Tone over time
# st.write("<br>",unsafe_allow_html=True)
# st.write("<center><h3>ESG Trend Over Time</h3></center>",
#          unsafe_allow_html=True)
# st.write("<br>",unsafe_allow_html=True)
col1,col2 = st.beta_columns((1,3))


###### RUN COMPUTATIONS WHEN A COMPANY IS SELECTED ######
#When a Company is slected show and run the following
company = st.selectbox("Select a Company to Analyze", companies)
date_slider = st.empty()
if company and company != "Select a Company":

    #Filter data to company
    df_company = df_data[df_data.Organization == company]


    #Create date widget
    start = df_company.DATE.min()
    end = df_company.DATE.max()
    selected_dates = st.date_input("Select a Date Range", value=[start,end],
                                   min_value=start, max_value=end, key=None)
    time.sleep(0.8)
    start, end = selected_dates

    #Filter company data to dates and esg categories
    df_company = filter_company_data(df_company, esg_categories, start, end)

    #Create List of publishers to filter in the sidebar
    publishers = df_company.SourceCommonName.sort_values().unique().tolist()
    publishers.insert(0, "all")
    publisher = st.sidebar.selectbox("Select Publisher", publishers)

    #Filter the company to the selected publisher
    df_company = filter_publisher(df_company, publisher)

    #Create Columns for our company time chart
    st.write("<br>", unsafe_allow_html=True)
    st.write("<center><h3>ESG Trend Over Time</h3></center>",
             unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    col1,col2 = st.beta_columns((1,3))


    #Create Altair Chart of EST Tone over Time
    line_metric = col1.radio("Choose Metric", options=["Tone", "NegativeTone",
        "PositiveTone", "Polarity", "ActivityDensity", "WordCount"])

    df_metr = df_company.groupby("DATE")[line_metric].mean().reset_index()
    df_metr.columns = ["DATE", line_metric]
    esg_tone_chart = alt.Chart(df_metr).mark_line(point=True).encode(
        x=alt.X('yearmonthdate(DATE):O', title='DATE'),
        y=line_metric,
        tooltip=['DATE',line_metric]
    ).interactive()
    col2.altair_chart(esg_tone_chart, use_container_width=True)


    #Expander to Revel Row Level Data
    URL_Expander = st.beta_expander(f"{company.title()} Data")
    URL_Expander.write(df_company[["DATE", "SourceCommonName", "URL", "Tone",
        "Polarity", "ActivityDensity", "SelfDensity", "WordCount"]])
    st.write("<br><br>", unsafe_allow_html=True)

    #Scatter plot of all articles for the time period
    c = alt.Chart(df_company, title="Article Tone").mark_circle().encode(
        x="NegativeTone:Q",
        y="PositiveTone:Q",
        size="WordCount:Q",
        color=alt.Color("Polarity:Q", scale=alt.Scale(scheme="purplered")),
        tooltip=[alt.Tooltip("Polarity", format=".3f"),
                 alt.Tooltip("NegativeTone", format=".3f"),
                 alt.Tooltip("PositiveTone", format=".3f"),
                 # alt.Tooltip("Tone", format=".3f"),
                 # alt.Tooltip("ActivityDensity", format=".2f"),
                 alt.Tooltip("DATE"),
                 alt.Tooltip("WordCount", format=",d"),
                 # "URL",
                 alt.Tooltip("SourceCommonName", title="Site")]
        ).interactive()
    st.altair_chart(c, use_container_width=True)


    num_neighbors = st.slider("Number of Connections", 1, 20, value=8)
    neighbor_cols = [f"n{i}_rec" for i in range(num_neighbors)]
    company_df = df_conn[df_conn.company == company]
    neighbors = company_df[neighbor_cols].iloc[0]


    ###### Horizontal Neighbor Similarity Chart ######
    neighbor_conf = pd.DataFrame({
        "Neighbor": neighbors,
        "Confidence": company_df[[f"n{i}_conf" for i in range(num_neighbors)]].values[0]
        })
    conf_plot = alt.Chart(neighbor_conf, title="Connected Companies"
                          ).mark_bar().encode(
        x="Confidence:Q",
        y=alt.Y("Neighbor:N", sort="-x"),
        tooltip=["Neighbor", alt.Tooltip("Confidence", format=".3f")],
        color=alt.Color("Confidence:Q", scale=alt.Scale(scheme="purplered")),
    ).configure_axis(grid=False)
    st.altair_chart(conf_plot, use_container_width=True)

    ###### EMBEDDING 3D PLOT ######
    # company_list = [company] + neighbors.to_list()
    color_f = lambda f: f"Company: {company.title()}" if f == company else (
        "Connected Company" if f in neighbors.values else "Other Company")
    embeddings["colorCode"] = embeddings.company.apply(color_f)
    print(embeddings.query("company=='visa'"))
    fig_3d = px.scatter_3d(embeddings, x="0", y="1", z="2", color='colorCode',
                           color_discrete_sequence=["lightgrey", "#C137A2",
                           "#694ED6"], opacity=0.5,
                           hover_data={"company": True, "colorCode": True,
                                       "0": False, "1": False, "2": False})
    st.plotly_chart(fig_3d, use_container_width=True)
