import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import time
from datetime import datetime
import pickle
import itertools
import plotly.express as px
from plot_setup import finastra_theme
from download_data import Data
#import networkx as nx
#import nx_altair as nxa
#from Graph import graph_creator


####### CACHED FUNCTIONS ######
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
def load_data(base_dir):
    data = Data().read("ten_days")
    companies = data["data"].Organization.sort_values().unique().tolist()
    companies.insert(0,"Select a Company")
    return data, companies


@st.cache(show_spinner=False,suppress_st_warning=True)
def filter_publisher(df_company,publisher):
    if publisher != 'all':
        df_company = df_company[df_company['SourceCommonName'] == publisher]
    return df_company


def get_melted_frame(data_dict, frame_names, keepcol=None, dropcol=None):
    if keepcol:
        reduced = {k: df[keepcol].rename(k) for k, df in data_dict.items()
                   if k in frame_names}
    else:
        reduced = {k: df.drop(columns=dropcol).mean(axis=1).rename(k)
                   for k, df in data_dict.items() if k in frame_names}
    df = (pd.concat(list(reduced.values()), axis=1).reset_index().melt("date")
            .sort_values("date").ffill())
    df.columns = ["DATE", "ESG", "Score"]
    return df


def filter_melted_date(df, start, end, date_col="DATE"):
    df = df[(df[date_col] >= pd.to_datetime(start)) &
            (df[date_col] <= pd.to_datetime(end))]
    return df

def main(start, end):
    ###### CUSTOMIZE COLOR THEME ######
    alt.themes.register("finastra", finastra_theme)
    alt.themes.enable("finastra")
    violet, fuchsia = ["#694ED6", "#C137A2"]


    ###### SET UP PAGE ######
    icon_path = "esg_ai_logo.png"
    st.set_page_config(page_title="ESG AI", page_icon=icon_path,
                       layout='centered', initial_sidebar_state='auto')
    _, logo, _ = st.beta_columns(3)
    logo.image(icon_path, width=200)
    style = ("text-align:center; padding: 0px; font-family: arial black;, "
             "font-size: 400%")
    title = f"<h1 style='{style}'>ESG<sup>AI</sup></h1><br><br>"
    st.write(title, unsafe_allow_html=True)


    ###### LOAD DATA ######
    base_path = os.path.join("Data", f"{start}_to_{end}")
    with st.spinner(text="Fetching Data..."):
        data, companies = load_data(base_path)
    df_conn = data["conn"]
    df_data = data["data"]
    embeddings = data["embed"]


    ####### CREATE SIDEBAR CATEGORY FILTER######
    st.sidebar.title("Filter Options")
    date_place = st.sidebar.empty()
    esg_categories = st.sidebar.multiselect("Select News Categories",
                                            ["E", "S", "G"], ["E", "S", "G"])






    ###### RUN COMPUTATIONS WHEN A COMPANY IS SELECTED ######
    company = st.selectbox("Select a Company to Analyze", companies)
    if company and company != "Select a Company":
        ###### FILTER ######
        df_company = df_data[df_data.Organization == company]
        diff_col = f"{company.replace(' ', '_')}_diff"
        esg_keys = ["E_score", "S_score", "G_score"]
        esg_df = get_melted_frame(data, esg_keys, keepcol=diff_col)
        ind_esg_df = get_melted_frame(data, esg_keys, dropcol="industry_tone")
        tone_df = get_melted_frame(data, ["overall_score"], keepcol=diff_col)
        ind_tone_df = get_melted_frame(data, ["overall_score"],
                                       dropcol="industry_tone")


        ###### DATE WIDGET ######
        start = df_company.DATE.min()
        end = df_company.DATE.max()
        # selected_dates = st.date_input("Select a Date Range", value=[start, end],
        #                                min_value=start, max_value=end, key=None)
        selected_dates = date_place.date_input("Select a Date Range",
            value=[start, end], min_value=start, max_value=end, key=None)
        time.sleep(0.8)  #Allow user some time to select the two dates -- hacky :D
        start, end = selected_dates


        ###### FILTER DATA ######
        df_company = filter_company_data(df_company, esg_categories, start, end)
        esg_df = filter_melted_date(esg_df, start, end)
        ind_esg_df = filter_melted_date(ind_esg_df, start, end)
        tone_df = filter_melted_date(tone_df, start, end)
        ind_tone_df = filter_melted_date(ind_tone_df, start, end)


        ###### PUBLISHER SELECT BOX ######
        publishers = df_company.SourceCommonName.sort_values().unique().tolist()
        publishers.insert(0, "all")
        publisher = st.sidebar.selectbox("Select Publisher", publishers)
        df_company = filter_publisher(df_company, publisher)


        ###### DISPLAY DATA ######
        URL_Expander = st.beta_expander(f"View {company.title()} Data:", True)
        URL_Expander.write(f"### {len(df_company):,d} Matching Articles for " +
                           company.title())
        display_cols = ["DATE", "SourceCommonName", "URL", "Tone", "Polarity",
                        "ActivityDensity", "SelfDensity"]  #  "WordCount"
        URL_Expander.write(df_company[display_cols])
        st.write("<br>", unsafe_allow_html=True)


        ###### CHART: METRIC OVER TIME ######
        st.write("<br>", unsafe_allow_html=True)
        col1, col2 = st.beta_columns((1, 3))

        metric_options = ["Tone", "NegativeTone", "PositiveTone", "Polarity",
                          "ActivityDensity", "WordCount", "Overall Score",
                          "ESG Scores"]
        line_metric = col1.radio("Choose Metric", options=metric_options)

        if line_metric == "ESG Scores":
            # Get ESG scores
            esg_df["WHO"] = company.title()
            ind_esg_df["WHO"] = "Industry Average"
            esg_plot_df = pd.concat([esg_df, ind_esg_df]).reset_index(drop=True)
            esg_plot_df.replace({"E_score": "Environment", "S_score": "Social",
                                 "G_score": "Governance"}, inplace=True)

            metric_chart = alt.Chart(esg_plot_df, title="Trends Over Time"
                                       ).mark_line().encode(
                x=alt.X("yearmonthdate(DATE):O", title="DATE"),
                y="Score:Q",
                # color=alt.Color("ESG:O", sort=None, legend=alt.Legend(
                #     orient="top", title=None)),
                color=alt.Color("ESG", sort=None, legend=alt.Legend(
                    title=None, orient="top")),
                strokeDash=alt.StrokeDash("WHO", sort=None, legend=alt.Legend(
                    title=None, symbolType="stroke", symbolFillColor="gray",
                    symbolStrokeWidth=4, orient="top")
                                          ),
                tooltip=["DATE", "ESG", alt.Tooltip("Score", format=".5f")]
                )

        else:
            if line_metric == "Overall Score":
                line_metric = "Score"
                tone_df["WHO"] = company.title()
                ind_tone_df["WHO"] = "Industry Average"
                plot_df = pd.concat([tone_df, ind_tone_df]).reset_index(drop=True)
            else:
                df1 = df_company.groupby("DATE")[line_metric].mean().reset_index()
                df2 = df_data.groupby("DATE")[line_metric].mean().reset_index()
                df1["WHO"] = company.title()
                df2["WHO"] = "Industry Average"
                plot_df = pd.concat([df1, df2]).reset_index(drop=True)
            metric_chart = alt.Chart(plot_df, title="Trends Over Time"
                                     ).mark_line().encode(
                x=alt.X("yearmonthdate(DATE):O", title="DATE"),
                y=f"{line_metric}:Q",
                color=alt.Color("WHO", legend=None),
                strokeDash=alt.StrokeDash("WHO", sort=None,
                    legend=alt.Legend(
                        title=None, symbolType="stroke", symbolFillColor="gray",
                        symbolStrokeWidth=4, orient="top",
                        ),
                    ),
                tooltip=["DATE", line_metric]
                )
        metric_chart = metric_chart.properties(height=340).interactive()
        col2.altair_chart(metric_chart, use_container_width=True)




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
            ).properties(
                height=350
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
            ).properties(
                height=450
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
            color=alt.Color("Confidence:Q", scale=alt.Scale(), legend=None)
        ).properties(
            height=40 * num_neighbors
        ).configure_axis(grid=False)
        st.altair_chart(conf_plot, use_container_width=True)


        ###### CHART: 3D EMBEDDING WITH NEIGHBORS ######
        color_f = lambda f: f"Company: {company.title()}" if f == company else (
            "Connected Company" if f in neighbors.values else "Other Company")
        embeddings["colorCode"] = embeddings.company.apply(color_f)
        point_colors = {company: violet, "Connected Company": fuchsia,
                        "Other Company": "lightgrey"}
        fig_3d = px.scatter_3d(embeddings, x="0", y="1", z="2",
                               color='colorCode',
                               color_discrete_map=point_colors,
                               opacity=0.4,
                               hover_name="company",
                               hover_data={c: False for c in embeddings.columns},
                               )
        fig_3d.update_layout(legend={"orientation": "h",
                                     "yanchor": "bottom",
                                     "title": None},
                             title={"text": "<b>Company Connections</b>",
                                    "x": 0.5, "y": 0.9,
                                    "xanchor": "center",
                                    "yanchor": "top",
                                    "font": {"family": "Futura", "size": 23}},
                             scene={"xaxis": {"visible": False},
                                    "yaxis": {"visible": False},
                                    "zaxis": {"visible": False}},
                             margin={"l": 0, "r": 0, "t": 0, "b": 0},
                             )
        st.plotly_chart(fig_3d, use_container_width=True)


        ###### CHART: ESG RADAR ######
        avg_esg = data["ESG"]
        avg_esg.rename(columns={"Unnamed: 0": "Type"}, inplace=True)
        avg_esg.replace({"T": "Overall", "E": "Environment",
                         "S": "Social", "G": "Governance"}, inplace=True)
        avg_esg["Industry Average"] = avg_esg.mean(axis=1)

        radar_df = avg_esg[["Type", company, "Industry Average"]].melt("Type")

        radar = px.line_polar(radar_df, r="value", theta="Type",
            color="variable", line_close=True, hover_name="Type",
            color_discrete_map={"Industry Average": fuchsia, company: violet})
        # radar.update_traces(fill="toself")
        radar.update_layout(template=None,
                            polar={
                                   # "bgcolor": "white",
                                   "radialaxis": {"showticklabels": False,
                                                  "ticks": ""},
                                   # "angularaxis": {"showticklabels":False,
                                   #                 "ticks": ""}
                                   })
        st.plotly_chart(radar, use_container_width=True)




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
    alt.themes.enable("default")

if __name__ == "__main__":
    start_date = "dec1"
    end_date = "dec10"
    main(start_date, end_date)
