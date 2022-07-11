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
  
def NW_Reg(h,x_train,y0_train,y1_train,x_test):
  '''
  Implementation of Nadaraya-Watson regression
  Inputs::
  h: Nadaraya regression bandwidth
  x_train: positions in pore where velocity in known (the sensors)
  y0_train: sensor velocity measurement in x direction
  y1_train: sensor velocity measurement in y direction
  x_test: positions in pore where velocity is unknown and to be estimated
  ''' 
  h += 1e-3 #adjust bandwidth
  y0_pred = np.zeros(x_test.shape[1])
  y1_pred = np.zeros(x_test.shape[1])
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
    ker_coef = ker_/(ker_.sum()) if ker_.sum() >0 else ker_
    y0_pred[i] = ker_coef.dot(y0_train)
    y1_pred[i] = ker_coef.dot(y1_train)

  return y0_pred, y1_pred
  
def LOOCV_NW_Reg(h,x_train,y0_train,y1_train):
  '''
  Implementation of Nadaraya-Watson regression with leave one out cross-validation
  h: bandwidth,
  x_train: cartesian location of point,
  y0_train: x velocity components,
  y1_train: y velocity components.
  '''
  h += 1e-3 #adjust bandwidth
  y0_pred = np.zeros(x_train.shape[1])
  y1_pred = np.zeros(x_train.shape[1])
  if len(x_train.shape) == 1:
    n = 1
  else:
    n = x_train.shape[0]
    x_tr = x_train.T
  for i,test in tqdm(enumerate(x_tr)):
    # print('==={i}==')
    #leave-one-out
    arr_x = np.delete(x_tr,i,axis = 0).T
    arr_y0 = np.delete(y0_train, i)
    arr_y1 = np.delete(y1_train, i)
    #NW estimation
    if n > 1:
      # print(arr_x.shape)
      ker0 = triangle_density(test[0]-arr_x[0],h)
      ker1 = triangle_density(test[1]-arr_x[1],h)
      ker_ = ker0 * ker1
    else:
      ker_ = triangle_density(test - arr_x,h)    
    ker_coef = ker_/(ker_.sum()) if ker_.sum() >0 else ker_
    y0_pred[i] = ker_coef.dot(arr_y0)
    y1_pred[i] = ker_coef.dot(arr_y1)

  return y0_pred, y1_pred
  
def Graph_NW_Reg(h,G,x_train,y0_train,y1_train,x_test):
  '''
  Implementation of graph Nadaraya-Watson regression.
  Inputs::
  G is the pore-pixel graph
  h is the bandwidth
  f is a contains two elements representing the velocity field in the x- and y-direction

  Returns:
  A list of velocity field prediction like input f
  '''
  # G_ = copy.deepcopy(G)
  h += 1e-3 #adjust bandwidth
  dist_dict = dict(nx.shortest_path_length(G)) #dictionary of shortest paths across the graph
  nodes_pos = node_pos(G)
  pred_f = [np.zeros_like(f[0]), np.zeros_like(f[1])]
  for key in tqdm(dist_dict.keys()):
    pos_ker = [] #list of node position and position-wise kernel density tuple
    key_dists = dist_dict[key]
    for neigh in key_dists:
      if key_dists[neigh] <= h:
        pos_ker.append((G.nodes[neigh]['pos'],1 - key_dists[neigh]/(h)))

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
    
    pred_f[0][G.nodes[key]['pos']] = vel_x/ker_sum if ker_sum >0 else 0
    pred_f[1][G.nodes[key]['pos']] = vel_y/ker_sum if ker_sum > 0 else 0
  return pred_f

def Graph_Node_NW_Reg(source,target,h,G,f,LOOCV = False): #dist_
  '''
  Implementation of graph Nadaraya-Watson regression node-wise with leave one out CV
  Inputs::
  source: source node
  target: a list of gauge nodes
  G is the pore-pixel graph
  h is the bandwidth
  f is a contains two elements representing the velocity field in the x- and y-direction
  LOOCV: Boolean. Turn on and off Leave-one-out cross validation
  dist_dict: dictionary of distances across graph

  Returns:
  A list of velocity field prediction like input f
  '''
  h += 1e-3 #adjust bandwidth
  ker_sum = 0
  vel_x = 0
  vel_y = 0
  # dist_ = {}
  if LOOCV:
    #leave one out cross validation
    target.remove(source)
  dist_ = nx.shortest_path_length(G,source=source)
  pos_ker = [] #list of node position and position-wise kernel density tuple
  for gauge in target:
    try:
      if dist_[gauge] <= h:
        pos_ker.append((G.nodes[gauge]['pos'],1 - dist_[gauge]/(h)))
    except:
      continue
  for pos,ker_coef in pos_ker:
    ker_sum += ker_coef
    vel_x += ker_coef * f[0][pos]
    vel_y += ker_coef * f[1][pos]

  if ker_sum == 0:
    return 0, 0
  return vel_x/ker_sum, vel_y/ker_sum