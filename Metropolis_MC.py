## Import libraries
import argparse
from pickletools import float8
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from tqdm import tqdm, trange
import wandb
import networkx as nx
import pandas as pd

from LS_Algorithm import get_nme

## Initialization and supplementary functions

def adj_matrix(G):
    '''
    creates adj and neigbors:
    adj: 2D adjacency matrix of a graph
    neighbors: list of lists of neighbors (indexes from 0 to N-1, where N is a number of nodes)
    '''
    adj = nx.to_numpy_array(G)
 
    neighbors = []
    for i in range(nx.number_of_nodes(G)):
        neighbors_ = [neighbor for neighbor in nx.neighbors(G, i)]
        neighbors.append(neighbors_)
    return adj, neighbors


def system_initialization(G, N_sensors):
    '''
    - Uses previously stated function adj_matrix to initialize 
    all needed information about graph
    - Initializes sensors positions
    ----------------------------------
    G: graph
    adj: adjacency matrix
    N_sensors: number of sensors
    sensors: ndarray of sensors' indexes 
    '''
    n = nx.number_of_edges(G)
    N_nodes = nx.number_of_nodes(G)

    adj, neighbors = adj_matrix(G)
    sensors = np.array([(N_nodes*i)//N_sensors for i in range(N_sensors)])

    return adj, neighbors, sensors

## Metropolis algorithm

def step(sensors, E_tot, Statistics, neighbors, adj):
    '''
    one step of a Metropolis algorithm cycle:

    1. shifts one random sensor to the neigboring node
    2. calculates the difference in energy
    3. accepts\rejects new location of the sensor
    '''

    # choose the random sensor
    choice = rng.choice
    old_sensor_number = choice(np.arange(N_sensors))
    old_node_number = sensors[old_sensor_number]

    # skip to the next step if there are no neighbors
    if len(neighbors[old_node_number]) == 0:
        # no neighbors
        Statistics[0] +=1 
        return sensors, E_tot, Statistics

    # choose the random neighboring node of the chosen one
    num_of_neighbors = len(neighbors[old_node_number])
    neighboring_loc = choice(np.arange(num_of_neighbors))
    new_node_number = neighbors[old_node_number][neighboring_loc]

    # skip to the next step if the chosen node is occupied
    if new_node_number in sensors:
        Statistics[1] +=1
        return sensors, E_tot, Statistics

    #shift the sensor
    # sensors_new = np.delete(sensors, old_sensor_number) 
    # sensors_new = np.append(sensors_new, new_node_number)
    sensors_new = sensors.copy()
    sensors_new[old_sensor_number] = new_node_number
    # calculate the difference in energy
    E_new = get_nme(sensors_new, ks, G)
    dE = E_new - E_tot
    # accept/reject
    dp = np.exp(-dE/T)
    rand = rng.random()
    wandb.log({'E/random': rand, 'E/acceptance probability': dp}, commit=False)
    if dp > rand:
        # accept
        E_tot = E_new
        Statistics[2] +=1
        return sensors_new, E_tot, Statistics
    else:
        # reject
        #return the sensor back
        Statistics[3] +=1
        return sensors, E_tot, Statistics

    
def cycle(sensors, E_tot, neighbors, adj, steps):
    '''
    cycle of metropolis algorithm's steps
    stores information about energy levels during the simulation
    '''

    Statistics = np.array([0.,  # 'no_neighbors'
                        0.,  # 'occupied'
                        0.,  # 'accepted'
                        0.]) # 'rejected'

    best_sensor_loc = sensors.copy()
    E_min = E_tot
    # best_step = 0


    for i in trange(1,steps+1):
        sensors, E_tot, Statistics = step(sensors, E_tot, 
                                          Statistics, neighbors, adj)
        if E_tot < E_min:
            best_sensor_loc = sensors.copy()
            
            E_min = E_tot
            # best_step = i
            wandb.log({'E/Emin': E_min, 'Best sensor location': best_sensor_loc}, commit=False)
        
        wandb.log({'E/E': E_tot, 'Statistics/no_neighbors':Statistics[0], 'Statistics/occupied':Statistics[1], 
                   'Statistics/accepted':Statistics[2], 'Statistics/rejected':Statistics[3]})
        
    Statistics /= steps
    return sensors, E_tot, Statistics,\
           best_sensor_loc, E_min

def simulation(adj, neighbors, sensors, steps):
    '''
    does the simulation and returns numerical results
    '''
    E_tot = get_nme(sensors, ks, G)
    sensors, E_tot, Statistics, best_sensor_loc, E_min =\
        cycle(sensors, E_tot, neighbors, adj, steps)
    print(f'E min: {E_min}\n best location:{best_sensor_loc}')
    # verification of right energy calculation
    E_fin = get_nme(sensors, ks, G)
    assert np.isclose(E_fin, E_tot), f'E_fin={E_fin}, E_tot={E_tot}'

    return sensors, Statistics, \
           best_sensor_loc, E_min

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--T',default=0.01, type=float)
    parser.add_argument('--N_sensors',default=5, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    ks = 2 #coarse_graining grid size
    G = nx.read_gpickle(f"pore_network_0{ks}.gpickle")
    rng = np.random.RandomState(42)
    args = init_parser()
    hyperparameter_defaults = dict(
        steps = 10**4
        )
    wandb.init(config=hyperparameter_defaults, entity="emmanuel-vsevolod")
    wandb.config.update(args)

    config = wandb.config
    T = config.T
    N_sensors = config.N_sensors
    steps = config.steps
    
    adj, neighbors, sensors = system_initialization(G, N_sensors)
    sensors, Statistics, best_sensor_loc, E_min =\
        simulation(adj, neighbors, sensors, steps)

    data = {"best_location":best_sensor_loc}
    df = pd.DataFrame(data)
    best_loc_table = wandb.Table(data=df)
    wandb.log({'best location table':best_loc_table}, commit=False)