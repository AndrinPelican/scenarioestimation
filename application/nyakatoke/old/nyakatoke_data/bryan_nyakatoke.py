# Main scientific computing modules
import numpy as np
import pandas as pd

# Import matplotlib
import matplotlib.pyplot as plt

# networkx module for the analysis of nyakatoke data
import networkx as nx
data = "./nyakatoke_data/"

Nyakatoke_ind = pd.read_stata(data+"Nyakatoke individual.dta")
Nyakatoke_dyad = pd.read_stata(data+"Nyakatoke dyadic.dta")
Nyakatoke_hh = pd.read_stata(data+"Nyakatoke household.dta")
Nyakatoke_dir = pd.read_stata(data+"Nyakatoke directed.dta")


Nyakatoke_ind.dropna(subset=['hhind'], inplace = True)               # Drop rows with missing identification number
Nyakatoke_ind['hh1']=Nyakatoke_ind['hhind'].apply(str).str[2:5]      # Create household ID number
Nyakatoke_ind['iid']=Nyakatoke_ind['hhind'].apply(str).str[5:7]      # Create individual ID number

# Find age of household head
Nyakatoke_ind['head_age'] = None
Nyakatoke_ind.loc[Nyakatoke_ind['iid']=='01', 'head_age'] = Nyakatoke_ind['age']

# Find sex of household head
gender_dict = {'male' : 1, 'female' : 0}
Nyakatoke_ind['head_sex'] = np.nan
Nyakatoke_ind.loc[Nyakatoke_ind['iid']=='01', 'head_sex'] = Nyakatoke_ind['sex'].map(gender_dict)

# Create completed primary school dummy variable
Nyakatoke_ind['primary'] = ((Nyakatoke_ind['education'] == 'finished primary') | (Nyakatoke_ind['education'] == 'secondary'))*1

hh_set_ind = set(Nyakatoke_ind['hh1'].unique())

# Create household-level dataset with age, sex and primary completion infomation
hh_var = Nyakatoke_ind[['head_age','head_sex','primary','hh1']].groupby('hh1').max()
hh_var['num_respondents'] = Nyakatoke_ind[['iid','hh1']].groupby('hh1').count()['iid']

hh_var.reset_index(inplace=True)

# Set of househoulds represented in household-level attribute data file
hh_set_hh_var = set(hh_var['hh1'].unique())

# Total number of activities engaged in by each individual
Nyakatoke_ind['act_total'] = Nyakatoke_ind['act_offfarm'] + Nyakatoke_ind['act_casual'] + Nyakatoke_ind['act_trade'] \
                             + Nyakatoke_ind['act_crop'] + Nyakatoke_ind['act_livest'] + Nyakatoke_ind['act_assets'] \
                             + Nyakatoke_ind['act_process']

# Compute totals by category for each household
hh_act = Nyakatoke_ind[['act_offfarm', 'act_casual', 'act_trade', 'act_crop', \
                        'act_livest', 'act_assets', 'act_process', 'act_total', 'hh1']].groupby('hh1').sum()

# Covert total counts into percentages as in De Weerdt (2004)
for activity in ['offfarm', 'casual', 'trade', 'crop', 'livest', 'assets', 'process']:
    hh_act['prc_' + activity] = (hh_act['act_' + activity] / hh_act['act_total']) * 100
    hh_act['prc_' + activity][
        hh_act['act_total'] == 0] = 0  # Avoid divide by zero error for households with no activity
    print(hh_act['act_' + activity].value_counts())
    hh_act.drop('act_' + activity, axis=1, inplace=True)

hh_act.drop('act_total', axis=1, inplace=True)
hh_act.reset_index(inplace=True)
print(hh_act.describe())

# Set of househoulds represented in household-level activity data file
hh_set_hh_act = set(hh_act['hh1'].unique())


Nyakatoke_dyad.loc[:,'hh1'] = Nyakatoke_dyad['hh1'].apply(str).str.zfill(5)

# remove decimal and digits to its right
Nyakatoke_dyad.loc[:,'hh1'] = Nyakatoke_dyad['hh1'].apply(str).str[0:3]

# Repeat conversions for hh2
Nyakatoke_dyad.loc[:,'hh2'] = Nyakatoke_dyad['hh2'].apply(str).str.zfill(5)
Nyakatoke_dyad.loc[:,'hh2'] = Nyakatoke_dyad['hh2'].apply(str).str[0:3]

Nyakatoke_dyad = Nyakatoke_dyad.merge(hh_var, on = 'hh1', how = 'left', copy = False)
Nyakatoke_dyad.rename(columns={'head_age' : 'head_age1', 'head_sex' : 'head_sex1', 'primary' : 'primary1', \
                               'num_respondents' : 'num_respondents1'}, inplace=True)

# Then for household 2 in the dyad
Nyakatoke_dyad = Nyakatoke_dyad.merge(hh_var, left_on='hh2', right_on = 'hh1', how = 'left', copy = False)

# NOTE: Last merge creates two instances of hh1 due to how the merge above was done. Drop the second instance and
#       rename the first
Nyakatoke_dyad.drop('hh1_y', axis=1, inplace=True)
Nyakatoke_dyad.rename(columns={'hh1_x' : 'hh1'}, inplace=True)

Nyakatoke_dyad.rename(columns={'head_age' : 'head_age2', 'head_sex' : 'head_sex2', 'primary' : 'primary2', \
                               'num_respondents' : 'num_respondents2'}, inplace=True)

Nyakatoke_dyad = Nyakatoke_dyad.merge(hh_act, on = 'hh1', how = 'left', copy = False)

for activity in ['offfarm', 'casual', 'trade', 'crop', 'livest', 'assets', 'process']:
    Nyakatoke_dyad.rename(columns={'prc_' + activity : 'prc_' + activity + '1'}, inplace=True)

# Then for household 2 in the dyad
Nyakatoke_dyad = Nyakatoke_dyad.merge(hh_act, left_on='hh2', right_on = 'hh1', how = 'left', copy = False)

# NOTE: Last merge creates two instances of hh1 due to how the merge above was done. Drop the second instance and
#       rename the first
Nyakatoke_dyad.drop('hh1_y', axis=1, inplace=True)
Nyakatoke_dyad.rename(columns={'hh1_x' : 'hh1'}, inplace=True)

for activity in ['offfarm', 'casual', 'trade', 'crop', 'livest', 'assets', 'process']:
    Nyakatoke_dyad.rename(columns={'prc_' + activity : 'prc_' + activity + '2'}, inplace=True)


# Comola and Fafchamps (2014) wealth formula
Nyakatoke_dyad['wealth1'] = (300000*Nyakatoke_dyad['land1'] + Nyakatoke_dyad['livestock1'])/100000
Nyakatoke_dyad['wealth2'] = (300000*Nyakatoke_dyad['land2'] + Nyakatoke_dyad['livestock2'])/100000

hh_set = set(Nyakatoke_dyad['hh1'].unique()) | set(Nyakatoke_dyad['hh2'].unique()) # Set of all households
N = len(hh_set)                                            # Number of households
n = N * (N - 1) //2                                        # Number of dyads

Nyakatoke_hh['clan'] = None
Nyakatoke_hh['religion'] = None
"""
for hh in hh_set:
    Nyakatoke_hh.loc[Nyakatoke_hh['hh'] == hh, 'clan'] = Nyakatoke_dyad.loc[Nyakatoke_dyad['hh1'] == hh, 'clan1']. \
            append(Nyakatoke_dyad.loc[Nyakatoke_dyad['hh2'] == hh, 'clan2'], ignore_index=True).unique()
    Nyakatoke_hh.loc[Nyakatoke_hh['hh'] == hh, 'religion'] = \
        Nyakatoke_dyad.loc[Nyakatoke_dyad['hh1'] == hh, 'religion1']. \
            append(Nyakatoke_dyad.loc[Nyakatoke_dyad['hh2'] == hh, 'religion2'], ignore_index=True).unique()
"""
G=nx.Graph()

# add households to graph object
hh_list = list(Nyakatoke_dyad['hh1'].unique())
G.add_nodes_from(hh_list)

# create edge list and add to graph
G.add_edges_from([(row['hh1'], row['hh2']) for index, row in Nyakatoke_dyad.iterrows() if \
                  (row['links']!='no link') ])

# add some household attributes to graph
nx.set_node_attributes(G, dict(zip(hh_var.hh1, hh_var.head_age)), 'age')
nx.set_node_attributes(G, dict(zip(hh_var.hh1, hh_var.primary)), 'education')

hh_wealth = {}
for index, row in Nyakatoke_dyad.iterrows():
    hh_wealth.setdefault(row['hh1'], []).append(row['wealth1'])
    hh_wealth.setdefault(row['hh2'], []).append(row['wealth2'])

hh_log_wealth = {}

for hh in hh_wealth:

    # Remove duplicate wealth measures
    hh_wealth[hh] = list(set(hh_wealth[hh]))[0]

    # Update log wealth dictionary
    hh_log_wealth.setdefault(hh, int(round(np.log(1 + hh_wealth[hh]) * 1000, 0)))

    # Catgorize into four groups
    if hh_wealth[hh] < 1.5:
        hh_wealth[hh] = 'very poor'
    elif (hh_wealth[hh] >= 1.5) & (hh_wealth[hh] < 3):
        hh_wealth[hh] = 'poor'
    elif (hh_wealth[hh] >= 3) & (hh_wealth[hh] < 6):
        hh_wealth[hh] = 'middle'
    elif hh_wealth[hh] >= 6:
        hh_wealth[hh] = 'rich'

# Attach wealth categorization to networkx graph
nx.set_node_attributes(G, hh_wealth, 'wealth')
nx.set_node_attributes(G, hh_log_wealth, 'log_wealth')

# put together a color map, one color for a category
# Berkeley Blue: #003262
# California Gold: #FDB515
# Golden Gate: #ED4E33
# Lawrence: #00B0DA
color_map = {'very poor': '#003262', 'poor': '#FDB515', 'middle': '#ED4E33', 'rich': '#00B0DA'}

# Construct degree attribute
nx.set_node_attributes(G, dict(nx.degree(G)), 'degree')



degree_sequence = pd.Series(sorted(dict(nx.degree(G)).values(),reverse=True)) # Degree sequence of nyakatoke as Pandas series
N               = len(degree_sequence)                                       # Number of households in the nyakatoke

# Complementary Cumulative Distribution Function for agent degrees
CCDF_degree = 1 - degree_sequence.value_counts().sort_index().cumsum()/N


ds = nx.eccentricity(G)

from collections import Counter
pl = dict(nx.shortest_path_length(G))   # Returns a dictionary of dictionaries
                                        # Each dictionary gives the path length from household i to all households j=1,..,N

shortest_path_length_counts = {}

for i, path_lengths_for_i in pl.items():
    for j, path_length in path_lengths_for_i.items():
        # If path_length not in dictionary add it as a key and set is item to zero (setdefault method)
        shortest_path_length_counts.setdefault(path_length, 0)
        shortest_path_length_counts[path_length] += 1

for path_length, count in shortest_path_length_counts.items():
    if path_length != 0:
        shortest_path_length_counts[path_length] = count // 2       # Divide by two to avoid double counting
    print(str(shortest_path_length_counts[path_length]) + " pairs of households are " + str(path_length) + " degrees apart")


empty=nx.Graph()                        # Empty triad
empty.add_nodes_from([1,2,3])

one_edge=nx.Graph()                     # One edge triad
one_edge.add_nodes_from(empty)
one_edge.add_edges_from([(1,2)])

two_star=nx.Graph()                     # Two star triad
two_star.add_nodes_from(empty)
two_star.add_edges_from([(1,2),(1,3)])

triangle=nx.Graph()                     # Triangle triad
triangle.add_nodes_from(empty)
triangle.add_edges_from([(1,2),(1,3),(2,3)])

# Position nodes in the shape of a triangle
node_pos = {1: [1,1], 2: [2,0], 3: [0,0]}

triad_census_fig = plt.figure(figsize=(6, 1.5))

# Empty triad
ax = triad_census_fig.add_subplot(1,4,1)
nx.draw_networkx(empty, pos=node_pos, with_labels=True, node_color='#FDB515', edge_color='#003262', width=3)
ax.axes.set_xlim([-0.5,2.5])
ax.axes.set_ylim([-0.5,1.5])
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('Empty')

# One edge triad
ax = triad_census_fig.add_subplot(1,4,2)
nx.draw_networkx(one_edge, pos=node_pos, with_labels=True, node_color='#FDB515', edge_color='#003262', width=3)
ax.axes.set_xlim([-0.5,2.5])
ax.axes.set_ylim([-0.5,1.5])
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('One edge')

# Two star triad
ax = triad_census_fig.add_subplot(1,4,3)
nx.draw_networkx(two_star, pos=node_pos, with_labels=True, node_color='#FDB515', edge_color='#003262', width=3)
ax.axes.set_xlim([-0.5,2.5])
ax.axes.set_ylim([-0.5,1.5])
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('Two star')

# Triangle triad
ax = triad_census_fig.add_subplot(1,4,4)
nx.draw_networkx(triangle, pos=node_pos, with_labels=True, node_color='#FDB515', edge_color='#003262', width=3)
ax.axes.set_xlim([-0.5,2.5])
ax.axes.set_ylim([-0.5,1.5])
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('Triangle')

# Save result
empty=nx.Graph()                        # Empty triad
empty.add_nodes_from([1,2,3])

one_edge=nx.Graph()                     # One edge triad
one_edge.add_nodes_from(empty)
one_edge.add_edges_from([(1,2)])

two_star=nx.Graph()                     # Two star triad
two_star.add_nodes_from(empty)
two_star.add_edges_from([(1,2),(1,3)])

triangle=nx.Graph()                     # Triangle triad
triangle.add_nodes_from(empty)
triangle.add_edges_from([(1,2),(1,3),(2,3)])

# Position nodes in the shape of a triangle
node_pos = {1: [1,1], 2: [2,0], 3: [0,0]}

triad_census_fig = plt.figure(figsize=(6, 1.5))

D = nx.adjacency_matrix(G) # Get Nyakatoke adjacency matrix as a Scipy sparse matrix object



print(" D is the adancy matrix wanted")

