import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time
import altair as alt
import pickle
import itertools
import networkx as nx
import nx_altair as nxa


st.set_page_config(page_title=None, page_icon="esg_ai_logo.png", layout='centered', initial_sidebar_state='auto')
#Article Filtering Categories for E, S, and G in the sidebar
esg_categories = st.sidebar.multiselect("Select News Categories",["E","S","G"],["E","S","G"])

####### Title ######
col1, col2,col3 = st.beta_columns((1,1,1))
# image = Image.open('esg_ai_logo.png')
# logo = col2.image(image,width=250)
col2.image("esg_ai_logo.png", width=250)
st.markdown("<hr></hr>",unsafe_allow_html=True)

###### LOAD DATA ######
@st.cache(show_spinner=False,suppress_st_warning=True)
def load_data(): 
    df_conn = pd.read_csv("Data/connectionsV2_10days.csv")
    df_data = pd.read_csv("Data/10daysample.csv",parse_dates=['DATE'],infer_datetime_format=True)
    df_data["DATE"] = df_data["DATE"].dt.date
    companies = df_data.Organization.sort_values(ascending=True).unique().tolist()
    companies.insert(0,"Select a Company")
    embeddings = pd.read_csv("Data/10_day_embeddings_pca.csv")
    return df_conn, df_data, companies, embeddings
df_conn, df_data,companies, embeddings = load_data()


#Filter companies by esg category and start and end date
@st.cache(show_spinner=False,suppress_st_warning=True)
def filter_company_data(df_company,esg_categories,start,end):
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
    line_metric = col1.radio("Choose Metric", options=["Tone", "NegativeTone",
        "PositiveTone", "Polarity", "ActivityDensity", "WordCount"])

    df_metr = pd.DataFrame(df_company.groupby("DATE")[line_metric].mean().reset_index(), columns=['DATE',line_metric])
    df_metr["DATE"] = pd.to_datetime(df_metr.DATE)
    esg_tone_chart = alt.Chart(df_metr).mark_line(point=True).encode(alt.X('yearmonthdate(DATE):O', title='DATE'),y=line_metric,tooltip=['DATE',line_metric]).interactive()
    col2.altair_chart(esg_tone_chart,use_container_width=True)


    #Expander to Revel Row Level Data
    URL_Expander = st.beta_expander(f"{company.title()} Data")
    URL_Expander.write(df_company[["DATE", "SourceCommonName", "URL", "Tone",
        "Polarity", "ActivityDensity", "SelfDensity", "WordCount"]])
    st.write("<br><br>",unsafe_allow_html=True)

    #Scatter plot of all articles for the time period
    c = alt.Chart(df_company).mark_circle().encode(
        x="NegativeTone:Q", 
        y="PositiveTone:Q", 
        size="WordCount:Q", 
        color=alt.Color("Polarity:Q", scale=alt.Scale(scheme="purplered")), 
        tooltip=["Tone", "NegativeTone", "PositiveTone", "Polarity", 
                 "WordCount", "ActivityDensity"]
        ).interactive()
    # c = alt.Chart(fake_df).mark_circle().encode(x='a', y='b', size='c', color='c', tooltip=['a', 'b','c'])
    st.altair_chart(c, use_container_width=True)



    ###### Network Graph ######
    G = nx.Graph()
    num_neighbors = st.slider("Number of Neighbors", 1, 20, value=8)
    neighbor_cols = ["company"] + [f"n{i}_rec" for i in range(num_neighbors)]
    company_df = df_conn[df_conn.company == company]
    neighbors = company_df[neighbor_cols]
    for org in neighbors.values[0]:
      G.add_node(org)
    df_X = df_data[df_data.Organization.isin(neighbors.values[0])]
    print(df_X.shape)
    df_edge = pd.DataFrame(df_X.groupby("URL").Organization.apply(list))
    print(df_edge.shape)
    df_edge = df_edge.reset_index()

    def get_tuples(row): 
      if len(row) > 1:
        return list(itertools.combinations(row,2))
      else: 
        return None

    def get_i(row,i): 
      return row[i]

    df_edge["SourceDest"] = df_edge.Organization.apply(lambda i: get_tuples(i))
    df_edge = df_edge.explode("SourceDest")
    df_edge = df_edge[~df_edge.SourceDest.isnull()]
    df_edge["Source"] = df_edge.SourceDest.apply(lambda i: get_i(i,0))
    df_edge["Dest"] = df_edge.SourceDest.apply(lambda i: get_i(i,1))
    df_edge = df_edge[["Source","Dest"]]
    edges = [tuple(r) for r in df_edge.to_numpy()]
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    viz = nxa.draw_networkx(G, pos=pos)
    st.altair_chart(viz.interactive(), use_container_width=True)


    neighbor_conf = pd.DataFrame({
        "Neighbor": company_df[[f"n{i}_rec" for i in range(num_neighbors)]].values[0],
        "Confidence": company_df[[f"n{i}_conf" for i in range(num_neighbors)]].values[0]
        })#.sort_values(by="Confidence", ascending=False)
    conf_plot = alt.Chart(neighbor_conf).mark_bar().encode(
        x=alt.X("Confidence:Q", sort="-x"),
        y="Neighbor:N",
        tooltip=["Confidence"]
    )
    st.altair_chart(conf_plot, use_container_width=True)
    # st.table(neighbor_conf)



    
    # edges = []
    # for i in G.edges: 
    #   if i[0] == (company): 
    #     edges.append(i)
    # G2 = G.edge_subgraph(edges)
    # pos = nx.spring_layout(G2)
    # viz = nxa.draw_networkx(G2, pos=pos, node_tooltip=["company:N"])
    # st.altair_chart(viz.interactive(), use_container_width=True)
    
