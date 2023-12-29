import numpy as np
import random
from scipy.linalg import block_diag

from monte_carlo_experiment.peer_effects.create_frendschip_network.create_random_geometric_graph import \
    random_geometric_graph

"""
Creates a peer nyakatoke where each agent has 5 friends from the 10 nodes closest to him


"""
def peer_network_5_neighbours(n_agents):

    adj_m = np.zeros((n_agents,n_agents))
    for i in range(n_agents):
        for _ in range(5):
            j = random.randint(i-5,i+5)
            j = (j+n_agents)%n_agents # get it in the right range
            adj_m[i,j]=1

    for i in range(n_agents):
        adj_m[i, i] = 0

    return adj_m

"""
Creates a peer nyakatoke where each agent has 5 friends from the 10 nodes closest to him


"""
def erdoes_reny_with_average_5_friends(n_agents):

    adj_m = np.zeros((n_agents,n_agents))
    for i in range(n_agents):
        for j in range(n_agents):
            if random.uniform(0,1)<(5/n_agents):
                adj_m[i, j] = 1

    # clear self loops:
    for i in range(n_agents):
        adj_m[i,i]= 0

    return adj_m


def erdoes_reny_with_average_5_friends_recipocal(n_agents):

    adj_m = np.zeros((n_agents,n_agents))
    for i in range(n_agents):
        for j in range(n_agents):
            if random.uniform(0,1)<(5/n_agents/2):
                adj_m[i, j] = 1
                adj_m[j, i] = 1

    # clear self loops:
    for i in range(n_agents):
        adj_m[i,i]= 0

    return adj_m




def erdoes_reny_with_average_5_friends_no_circle(n_agents):

    adj_m = np.zeros((n_agents,n_agents))
    for i in range(n_agents):
        for j in range(i,n_agents):
            if random.uniform(0,1)<(5/n_agents*2):
                adj_m[j, i] = 1

    # clear self loops:
    for i in range(n_agents):
        adj_m[i,i]= 0

    return adj_m


"""

Creates a peer nyakatoke where each agent has 5 friends from the 10 nodes closest to him


"""
def block_marix_full(n_agents, n_blocks):


    assert 0 == n_agents % n_blocks
    single_matrixes = [np.ones((int(n_agents / n_blocks),int(n_agents / n_blocks))) for i in range(n_blocks)]
    block_matrix = block_diag(*single_matrixes)

    # clear self loops:
    for i in range(n_agents):
        block_matrix[i,i]= 0

    return block_matrix


def block_erdös_reny(n_agents, n_blocks):

    assert 0 == n_agents%n_blocks
    single_matrixes = [erdoes_reny_with_average_5_friends(int(n_agents/n_blocks)) for i in range(n_blocks)]
    block_matrix = block_diag(*single_matrixes)

    return block_matrix

def block_erdös_reny_reciprocal(n_agents, n_blocks):

    assert 0 == n_agents%n_blocks
    single_matrixes = [erdoes_reny_with_average_5_friends_recipocal(int(n_agents/n_blocks)) for i in range(n_blocks)]
    block_matrix = block_diag(*single_matrixes)
    return block_matrix


def block_erdös_reny_no_cycles(n_agents, n_blocks):

    assert 0 == n_agents%n_blocks
    single_matrixes = [erdoes_reny_with_average_5_friends_no_circle(int(n_agents/n_blocks)) for i in range(n_blocks)]
    block_matrix = block_diag(*single_matrixes)
    return block_matrix



def block_random_geometric_graph(n_agents, n_blocks):
    assert 0 == n_agents % n_blocks
    single_matrixes = [random_geometric_graph(int(n_agents / n_blocks)) for i in range(n_blocks)]
    block_matrix = block_diag(*single_matrixes)
    return block_matrix



