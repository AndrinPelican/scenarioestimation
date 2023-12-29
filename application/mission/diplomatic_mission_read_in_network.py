import pickle
import networkx as nx
import numpy as np



def read_in_correlates_of_war():
    with open('diplomatic_americas_in_2005.gpkl', 'rb') as f:
        di_graphx = pickle.load(f)

    node_list = list(di_graphx.nodes)
    adj_m = nx.to_numpy_array(di_graphx, node_list)
    node_to_country_name = nx.get_node_attributes(di_graphx,"iso3")
    node_to_region = nx.get_node_attributes(di_graphx,"region")

    north_america = ["USA", "CAN", "MEX"]
    middle_america = ["BHS", "BLZ", "GTM", "HND", "SLV", "NIC", "CRI", "PAN"]
    south_america = ["VEN", "GUY", "SUR", "ECU", "PER", "BRA", "BOL", "PRY", "CHL", "ARG", "URY"]
    islands = ["BHS", "CUB", "HTI", "DOM", "JAM", "TTO", "BRB", "DMA", "GRD", "LCA", "VCT", "ATG", "KNA"]


    # creating the node variables which determine the groups
    var_dict = {}
    for i in range(adj_m.shape[0]):

        var_dict[i] = {}
        node = node_list[i]

        if node_to_country_name[node] in set(south_america):
            var_dict[i]["geo"] = 0
        elif node_to_country_name[node] in set(islands):
            var_dict[i]["geo"] = 1
        elif node_to_country_name[node] in set(middle_america):
            var_dict[i]["geo"] = 2
        else:
            var_dict[i]["geo"] = 3 # north america mainland

    return adj_m, var_dict

adj_m_correlates_of_war, var_dict_correlates_of_war = read_in_correlates_of_war()

