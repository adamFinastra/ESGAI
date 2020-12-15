import pandas as import pd
import numpy as np
import networkx as nx
import itertools

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
