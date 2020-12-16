import streamlit as st
import pandas as pd
from datetime import datetime
import time
import altair as alt
import pickle
import itertools
import plotly.express as px
from plot_setup import finastra_theme
import numpy as np
#import networkx as nx
#import nx_altair as nxa
#from Graph import graph_creator


####### CACHED FUNCTIONS ######
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


@st.cache(show_spinner=False,suppress_st_warning=True)
def filter_publisher(df_company,publisher):
    if publisher != 'all':
        df_company = df_company[df_company['SourceCommonName'] == publisher]
    return df_company


###### CUSTOMIZE COLOR THEME ######
# alt.themes.register("finastra", finastra_theme)
# alt.themes.enable("finastra")
# print(alt.themes.names())


###### SET UP PAGE ######
icon_path = "esg_ai_logo.png"
st.set_page_config(page_title="ESG AI", page_icon=icon_path,
                   layout='centered', initial_sidebar_state='auto')
violet, fuchsia = ["#694ED6", "#C137A2"]
_, logo, _ = st.beta_columns(3)
logo.image(icon_path, width=200)
style = ("text-align:center; padding: 0px; font-family: arial black;, "
         "font-size: 400%")
title = f"<h1 style='{style}'>ESG<sup>AI</sup></h1><br><br>"
st.write(title, unsafe_allow_html=True)


###### LOAD DATA ######
df_conn, df_data, companies, embeddings = load_data()


####### CREATE SIDEBAR CATEGORY FILTER######
esg_categories = st.sidebar.multiselect("Select News Categories",
                                        ["E", "S", "G"], ["E", "S", "G"])


###### RUN COMPUTATIONS WHEN A COMPANY IS SELECTED ######
company = st.selectbox("Select a Company to Analyze", companies)
if company and company != "Select a Company":
    ###### FILTER ######
    df_company = df_data[df_data.Organization == company]


    ###### DATE WIDGET ######
    start = df_company.DATE.min()
    end = df_company.DATE.max()
    selected_dates = st.date_input("Select a Date Range", value=[start, end],
                                   min_value=start, max_value=end, key=None)
    time.sleep(0.8)  #Allow user some time to select the two dates -- hacky :D
    start, end = selected_dates


    ###### FILTER DATA ######
    df_company = filter_company_data(df_company, esg_categories, start, end)


    ###### PUBLISHER SELECT BOX ######
    publishers = df_company.SourceCommonName.sort_values().unique().tolist()
    publishers.insert(0, "all")
    publisher = st.sidebar.selectbox("Select Publisher", publishers)
    df_company = filter_publisher(df_company, publisher)


    ###### DISPLAY DATA ######
    URL_Expander = st.beta_expander(f"View {company.title()} Data:", True)
    display_cols = ["DATE", "SourceCommonName", "URL", "Tone", "Polarity",
                    "ActivityDensity", "SelfDensity"]  #  "WordCount"
    URL_Expander.write(df_company[display_cols])
    st.write("<br>", unsafe_allow_html=True)


    ###### CHART: METRIC OVER TIME ######
    st.write("<br>", unsafe_allow_html=True)
    st.write("<center><h3>ESG Trend Over Time</h3></center>",
             unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    col1, col2 = st.beta_columns((1, 3))

    metric_options = ["Tone", "NegativeTone", "PositiveTone", "Polarity",
                      "ActivityDensity", "WordCount"]
    line_metric = col1.radio("Choose Metric", options=metric_options)

    df_metr = df_company.groupby("DATE")[line_metric].mean().reset_index()
    df_metr.columns = ["DATE", line_metric]

    esg_tone_chart = alt.Chart(df_metr).mark_line(point=True, #color=fuchsia
                        ).encode(
        x=alt.X("yearmonthdate(DATE):O", title="DATE"),
        y=line_metric,
        tooltip=["DATE", alt.Tooltip(line_metric, format=".3f"),]
    ).interactive()
    col2.altair_chart(esg_tone_chart, use_container_width=True)

    extra_tone_options = ["Show Connection Avg", "Show Index Avg"]


    ###### CHART: HEATMAP OF DOCUMENT WITH FEATURES #####
    heatmap_Expander = st.beta_expander("Document Tone  Analyzer", False)
    df_heat = df_company[["URL","Tone"]]
    df_heat["Tone"] = np.log2(df_heat.Tone)
    df_heat = df_heat.melt(id_vars=["URL"]).sort_values(by='variable',ascending=False)
    heatmap = alt.Chart(df_heat).mark_rect().encode(
        x=alt.X('variable:N', title="Document Tone"),
        y=alt.Y('URL:N', title="URL"),
        color=alt.Color('value:Q', title="Tone Intensity"),
        tooltip=["URL"]
        ).interactive()
    heatmap_Expander.altair_chart(heatmap, use_container_width=True)


    ###### CHART: SCATTER OF ARTICLES OVER TIME #####
    scatter = alt.Chart(df_company, title="Article Tone").mark_circle().encode(
        x="NegativeTone:Q",
        y="PositiveTone:Q",
        size="WordCount:Q",
        color=alt.Color("Polarity:Q", scale=alt.Scale()),
        tooltip=[alt.Tooltip("Polarity", format=".3f"),
                 alt.Tooltip("NegativeTone", format=".3f"),
                 alt.Tooltip("PositiveTone", format=".3f"),
                 alt.Tooltip("DATE"),
                 alt.Tooltip("WordCount", format=",d"),
                 alt.Tooltip("SourceCommonName", title="Site")]
        ).interactive()
    st.altair_chart(scatter, use_container_width=True)


    ###### NUMBER OF NEIGHBORS TO FIND ######
    num_neighbors = st.slider("Number of Connections", 1, 20, value=8)
    neighbor_cols = [f"n{i}_rec" for i in range(num_neighbors)]
    company_df = df_conn[df_conn.company == company]
    neighbors = company_df[neighbor_cols].iloc[0]


    ###### CHART: NEIGHBOR SIMILIARITY ######
    neighbor_conf = pd.DataFrame({
        "Neighbor": neighbors,
        "Confidence": company_df[[f"n{i}_conf" for i in
                                  range(num_neighbors)]].values[0]})
    conf_plot = alt.Chart(neighbor_conf, title="Connected Companies"
                          ).mark_bar().encode(
        x="Confidence:Q",
        y=alt.Y("Neighbor:N", sort="-x"),
        tooltip=["Neighbor", alt.Tooltip("Confidence", format=".3f")],
        color=alt.Color("Confidence:Q", scale=alt.Scale()),
    ).configure_axis(grid=False)
    st.altair_chart(conf_plot, use_container_width=True)


    ###### CHART: 3D EMBEDDING WITH NEIGHBORS ######
    color_f = lambda f: f"Company: {company.title()}" if f == company else (
        "Connected Company" if f in neighbors.values else "Other Company")
    embeddings["colorCode"] = embeddings.company.apply(color_f)
    fig_3d = px.scatter_3d(embeddings, x="0", y="1", z="2", color='colorCode',
                           color_discrete_sequence=["lightgrey", fuchsia,
                           violet], opacity=0.5,
                           hover_data={"company": True, "colorCode": True,
                                       "0": False, "1": False, "2": False})
    st.plotly_chart(fig_3d, use_container_width=True)



    ###### CONNECTION HISTOGRAM WITH ESG SCORE/METRIC ######
    # source = pd.DataFrame({
    # 'Trial A': np.random.normal(0, 0.8, 1000),
    # 'Trial B': np.random.normal(-2, 1, 1000),
    # 'Trial C': np.random.normal(3, 2, 1000)
    # })
    #
    # layered_hist = alt.Chart(source).transform_fold(
    #     ['Trial A', 'Trial B', 'Trial C'],
    #     as_=['Experiment', 'Measurement']
    # ).mark_area(
    #     opacity=0.3,
    #     interpolate='step'
    # ).encode(
    #     alt.X('Measurement:Q', bin=alt.Bin(maxbins=100)),
    #     alt.Y('count()', stack=None),
    #     alt.Color('Experiment:N')
    # ).interactive()
    # st.altair_chart(layered_hist,use_container_width=True)
