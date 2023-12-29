import csv
import numpy as np

def import_csv_removing_missing_values(filename):
    indexes_to_drop = ['122', '063', '047', '091']
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['ego'] not in indexes_to_drop and row['alter'] not in indexes_to_drop:
                data.append(row)
    return data

def get_flatten_ind(id_from, id_to, num_nodes):
    # get the right index to write it in, considering the 0 os in the adj matrix
    shift_do_consider_diag = + -1 if id_to > id_from else 0
    k = id_from * (num_nodes - 1) + id_to + shift_do_consider_diag
    return k

def load_nyakatoke_network(use_sender_reciver_effect = False, s_function=None):
    # Usage example
    filename = 'NyakatokeEstimationSample.csv'
    csv_data = import_csv_removing_missing_values(filename)

    ego_list = []
    alter_list = []
    link_list = []
    for row in csv_data:
        ego_list.append(row['ego'])
        alter_list.append(row['alter'])
        link_list.append(row['link'])

    # Get unique nodes from ego and alter columns
    nodes = sorted(list(set(ego_list + alter_list)))
    num_nodes = len(nodes)

    # adjacency matrix
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for ego, alter, link in zip(ego_list, alter_list, link_list):
        ego_index = nodes.index(ego)
        alter_index = nodes.index(alter)
        adjacency_matrix[ego_index, alter_index] = 1 if link == 'True' else 0

    variables = ['kinship_pcs', 'kinship_nnuacgg', 'kinship_other', 'distance', 'same_religion', 'same_clan', 'prim_i_X_prim_j', 'activity_overlap', 'age_difference', 'wealth_difference']
    X = np.zeros((len(csv_data) ,len(variables)))
    Y = np.zeros((len(csv_data)))
    for i, row in enumerate(csv_data):
        ind_from = nodes.index(row['ego'])
        ind_to = nodes.index(row['alter'])
        k = get_flatten_ind(ind_from, ind_to,num_nodes)

        for j, variable in enumerate(variables):
            value = row[variable]
            if (variable=='distance'):
                value = float(value)/1000 # transform to kilometers


            if (variable=='activity_overlap'):
                value = float(value)/10 # transform to kilometers


            if (variable=='age_difference'):
                value = float(value)/10 # transform to kilometers



            if value == 'True':
                X[k, j] = 1.
            elif value == 'False':
                X[k, j] = 0.
            else:
                X[k, j] = float(value)

        if row['link'] == 'True':
            Y[k] = 1.
        else:
            Y[k] = 0.


    # constant or Sender reciver effect
    if use_sender_reciver_effect:
        X_2 = np.zeros((len(csv_data) ,num_nodes*2))
        for i, row in enumerate(csv_data):
            ind_from = nodes.index(row['ego'])
            ind_to = nodes.index(row['alter'])
            k = get_flatten_ind(ind_from, ind_to,num_nodes)
            X_2[k, ind_from] += 1.
            X_2[k, ind_to + num_nodes] += 1.
        variables += ["fe_"+str(i) for i in range(num_nodes*2-1)]
        X = np.concatenate([X, X_2[:,1:]], axis=1) # start from 1 to avoid mulitcoliniarty
    else:
        variables += ["constant"]
        X = np.concatenate([X, np.ones((len(csv_data),1))], axis=1)  # start from 1 to avoid mulitcoliniarty

    # the endoginious variable
    if s_function!=None:
        print("adding the endogenious varible in the regression")
        Y_In_X = np.zeros((len(csv_data),1))
        matrix_of_effects = s_function(adjacency_matrix)

        for i, row in enumerate(csv_data):
            ind_from = nodes.index(row['ego'])
            ind_to = nodes.index(row['alter'])
            k = get_flatten_ind(ind_from, ind_to,num_nodes)
            Y_In_X[k, 0] += matrix_of_effects[ind_from, ind_to]

    return adjacency_matrix, variables, X, Y, Y_In_X


