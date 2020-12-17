# Databricks notebook source
# DBTITLE 1,Imports
import pandas as pd
import numpy as np
import networkx as nx
from nodevectors import Node2Vec as NVVV
import os
import itertools
from sklearn.decomposition import PCA

# COMMAND ----------

# DBTITLE 1,Load Data
def load_data(save_path,file_name): 
  df = (spark.read.format("delta")
                      .option("header", "true")
                      .option("inferSchema", "true")
                      .load(os.path.join(save_path, file_name))
           )

  df = df.toPandas()
  return df

file_name = "<LOAD FILE HERE>"
df = pd.read_csv(file_name)

# COMMAND ----------

# DBTITLE 1,Filter Out Non ESG related URLs
def filter_non_esg(df): 
  return df[(df['E']==True) | (df['S'] == True) | (df['G'] == True)]
df2 = filter_non_esg(df)

# COMMAND ----------

# DBTITLE 1,Get List of Organizations
organizations = list(set(df_edge.Source.tolist()).union(set(df_edge.Dest.tolist())))

# COMMAND ----------

# DBTITLE 1,Class to create Graph 
class graph_creator: 
  def __init__(df2):
    self.df2 = df2
  
  def create_graph(self): 
    df_edge = pd.DataFrame(self.df2.groupby("URL").Organization.apply(list))
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
    map = df_edge.groupby(['Source', 'Dest']).size()
    def get_weight(row,map): 
      return map[row.Source,row.Dest]
    df_edge["weight"] = df_edge[["Source","Dest"]].apply(lambda i: get_weight(i,map),axis=1)
    self.organizations = list(set(df_edge.Source.tolist()).union(set(df_edge.Dest.tolist())))
    self.G = nx.from_pandas_edgelist(df_edge, 'Source', 'Dest',
                                create_using=nx.Graph(), edge_attr='weight')
    return G

# COMMAND ----------

# DBTITLE 1,Create Graph
g = graph_creator(df2)
G = g.create_graph()

# COMMAND ----------

# DBTITLE 1,Train Node2Vec
g2v = NVVV()
g2v.fit(G)

# COMMAND ----------

# DBTITLE 1,Dimensionality Reduction to 3D Space From 32D Embeddings
embeddings = g2v.model.wv.vectors
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(embeddings)
d_e = pd.DataFrame(principalComponents)
d_e['company'] = organizations
embedding_fp = "<FILE PATH FOR EMBEDDINGS>"
d_e.to_csv(embedding_fp,index=None)

# COMMAND ----------

# DBTITLE 1,Get Top 25 Connections from Cosine Similarity from Embeddings
def expand_tuple(row): 
  return row[0],row[1]

l = []
for i in organizations:
  sim = g2v.model.wv.most_similar(i,topn=25)
  l.append(sim)
c = [f"n{i}" for i in range(25)]
df_sim = pd.DataFrame(l,columns=c)
df_sim["company"] = organizations
for i in c: 
  cols = [i+"_rec",i+"_conf"]
  df_sim[cols] = df_sim[i].apply(pd.Series)
df_sim = df_sim.drop(c,axis=1)
connections_fp = "<CONNECTIONS FILEPATH HERE>"
df_sim.to_csv(connections_fp,index=None)
