



"""

This script generates random geometric graphs according to:


http://bryangraham.github.io/econometrics/downloads/working_papers/DynamicNetworks/Homophily_and_Transitivity_April2016.pdf


The agents are uniformly scatter in a sqrt(n_agents) field

They form a link with probability 0.75 if the other agents is within radius r



"""


r = 1.3 # average 4 degree
link_prob = 0.75


r = 1.7# average 4 degree
link_prob = 0.5


import random
import numpy as np


def random_geometric_graph(n_agents, sort = False):

    adj_m = np.zeros((n_agents,n_agents))

    locations = random_locations_of_agents(n_agents)
    if sort:
        locations = sort_locations(locations)

    for i in range(n_agents):
        for j in range(n_agents):
            adj_m[i,j] = random_link_according_distance(i,j, locations)

    return adj_m



def random_locations_of_agents(n_agents):

    locations = []
    for i in range(n_agents):
        x = random.uniform(0,np.sqrt(n_agents))
        y = random.uniform(0,np.sqrt(n_agents))
        locations.append([x,y])

    return locations



def random_link_according_distance(i,j, locations):

    dist = distance(locations[i], locations[j])

    # random link if close enough and not a self loop
    if random.uniform(0,1)<link_prob and dist<r and i!=j:
        return 1
    else:
        return 0


def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def sort_locations(sort_locations):
    return sorted(sort_locations)
