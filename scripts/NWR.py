import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
import PIL
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
from tqdm.notebook import tqdm
import json
import time
import cv2
import random
import glob
import copy
from sklearn.metrics import r2_score

#triangle kernel function
tri_f = lambda x,h: 1 - abs(x)/h if 1 - abs(x)/h >= 0 else 0

def triangle_density(x,h):
    '''
    triangle kernel density
    Inputs::
    x: position variable (in this study, this represents difference in location)
    h: kernel bandwidth
    '''
  density_array = np.ones_like(x)
  for i,xx in enumerate(x):
    density_array[i] = tri_f(xx,h)
  return density_array

def NW_reg(h,x_train,y0_train,y1_train,x_test):
    '''
    Implementation of Nadaraya-Watson regression
    Inputs::
    h: Nadaraya regression bandwidth
    x_train: positions in pore where velocity in known (the sensors)
    y0_train: sensor velocity measurement in x direction
    y1_train: sensor velocity measurement in y direction
    x_test: positions in pore where velocity is unknown and to be estimated
    '''
  y0_pred = np.zeros_like(y0_train)
  y1_pred = np.zeros_like(y1_train)
  if len(x_train.shape) == 1:
    n = 1
  else:
    n = x_train.shape[0]
    x_test = x_test.T
  for i,test in tqdm(enumerate(x_test)):
    # print('==={i}==')
    if n > 1:
      ker0 = triangle_density(test[0]-x_train[0],h)
      ker1 = triangle_density(test[1]-x_train[1],h)
      ker_ = ker0 * ker1
    else:
      ker_ = triangle_density(test - x_train,h)
    ker_coef = ker_/(ker_.sum())
    y0_pred[i] = ker_coef.dot(y0_train)
    y1_pred[i] = ker_coef.dot(y1_train)

  return y0_pred, y1_pred

def Graph_NW_reg(h,G,f):
  '''
  Implementation of graph Nadaraya-Watson regression.
  Inputs:
  G is the pore-pixel graph
  h is the bandwidth
  f is a contains two elements representing the velocity field in the x- and y-direction

  Returns:
  A list of velocity field prediction like input f
  '''
  # G_ = copy.deepcopy(G)
  dist_dict = dict(nx.shortest_path_length(G)) #dictionary of shortest paths across the graph
  nodes_pos = node_pos(G)
  pred_f = [np.zeros_like(f[0]), np.zeros_like(f[1])]
  for key in tqdm(dist_dict.keys()):
    pos_ker = [] #list of node position and position-wise kernel density tuple
    key_dists = dist_dict[key]
    for neigh in key_dists:
      if key_dists[neigh] <= h+1:
        pos_ker.append((G.nodes[neigh]['pos'],1 - key_dists[neigh]/(h+1)))

    ker_sum = 0
    vel_x = 0
    vel_y = 0
    # f_X = f[:,:,0]
    # f_Y = f[:,:,1]
    for pos,ker_coef in pos_ker:
      ker_sum += ker_coef
      vel_x += ker_coef * f[0][pos]
      vel_y += ker_coef * f[1][pos]
    # print(G.nodes[key]['pos'],vel_x/ker_sum, vel_y/ker_sum)
    
    pred_f[0][G.nodes[key]['pos']] = vel_x/ker_sum
    pred_f[1][G.nodes[key]['pos']] = vel_y/ker_sum
  return pred_f