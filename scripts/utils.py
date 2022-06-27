#####functions
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
# from torchinfo import summary
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

#functions written for this research project
#rotate image 90 degrees clockwise
def show_pore(A):
  plt.imshow(np.rot90(A), cmap='Greys')
  plt.axis('off');
  plt.show()
  return

#pixel to graph
def pix_to_graph(A, kernel_size=1, scaled = False, flip = False, rotate = True):
  '''
  inputs: takes 2 inputs
  - pore array, and
  - kernel_size; for coarse-graining... default set to 1 if there is no coarse graining.
  - Scaled: This determines where the node position features stores the coarse grained position or a kernel size enlarge versio
  
  outputs: returns 4 outputs
  -a pore network whose nodes stors their location, 
  - number of nodes in the network,
  - number of edges in the network, and
  - number of clusters the network has.
  '''
  if flip:
    A = np.flip(A,1) #flip to aid visualization
  if rotate:
    A = np.rot90(A)
  G_T = nx.Graph()
  k,l = A.shape
  for i in range(k):
    for j in range(l):
      if A[i,j] <= kernel_size**2/2:
        #add node and save node's position
        if not scaled:
          G_T.add_node(i*l+j,pos = (i,j))
        else:
          G_T.add_node(i*l+j,pos = (kernel_size*i,kernel_size*j))
        for coord in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]: #check 4 neighbours
          if 0<=coord[0]<k and 0<=coord[1]<l:
            if A[coord[0],coord[1]] <= kernel_size**2/2:
              #add node and save node's position
              if not scaled:
                G_T.add_node(coord[0]*l+coord[1],pos = (coord[0],coord[1]))
              else:
                G_T.add_node(coord[0]*l+coord[1],pos = (kernel_size*coord[0],kernel_size*coord[1]))
              #add edge
              G_T.add_edge(i*l+j,coord[0]*l+coord[1])
  return G_T, G_T.number_of_nodes(), G_T.number_of_edges(), len(list(nx.connected_components(G_T)))

def node_pos(G_T):
  '''
  inputs: takes the pore network
  outputs: return a dictionary whose keys are the node identity 
  nd values are their respective locations.
  '''
  node_pos= {}
  for node in G_T.nodes:
    node_pos[node] = list(G_T.nodes[node].values())[0]
  return node_pos

def coarse_grain(img_array,kernel_size):
  #only square images are supported
  if img_array.shape[0]%kernel_size == 0: #and img_array[1]%kernel_size == 0:
    m = nn.Conv2d(1,1,kernel_size=kernel_size,stride= kernel_size,bias=False)
  else:
    #padding required if image size is indivisible by kernel size
    pad = int((kernel_size - img_array.shape[0]%kernel_size)/2)
    m = nn.Conv2d(1,1,kernel_size=kernel_size,stride= kernel_size,bias=False, padding=pad)
  weights = torch.ones_like(m.weight)
  with torch.no_grad():
      m.weight = nn.Parameter(weights)
  img_array = torch.from_numpy(img_array.copy()).reshape((1,1)+img_array.shape)
  CG= m(img_array.float()).detach().numpy().squeeze()
  return CG

def porenet_image(A,kernel_size=1, s=15, flip = True, rotate = False):
  '''
  inputs: takes three imputs
  - A: The pore array,
  - kernel_size: size of kernel for coarse graining operation, and
  - s: The figure size... default set to 15
  outputs: visualization of pore network on image
  '''
  #construct graph
  if kernel_size == 1:
    G_T,n,e,c = pix_to_graph(A, flip = flip, rotate = rotate)
  else:
    G_T,n,e,c = pix_to_graph(coarse_grain(A,kernel_size),kernel_size, scaled = True, flip = flip, rotate = rotate)
  print(f'{n} nodes, {e} edges, and {c} clusters')
  #visualization
  fig = plt.figure(figsize=(s,s))
  plt.imshow(np.rot90(A), cmap='Greys')
  nx.draw(G_T,pos = node_pos(G_T))
  # ax.axis("off");
  plt.show()
  return G_T

def get_total_space(array):
    '''
    Input
    array: binary structure array of porous/complex structure with 0 representing the free space and 1 representing the solids
    Returns
    A list of indices of spaces in porous/complex structure
    '''
    output = []
    for i in np.transpose(np.nonzero(array==0)):
        output.append(tuple(i))
    return output

#choose kernel size for coarse-graining
def data_process(pore_array,ks, vel_dir):
  '''
  Takes: 
  pore_array: array of pore space
  ks: coarse-graining (kernel) size,
  vel_dir : path to velocity profile and
  Returns:
  f : velocity data, 
  CG_G : graph (from coarse graining), 
  CG_f : [CG_f_0,CG_f_1]; a list of coarse-grained velocity data, 
  x : position data for training non-geometric method,
  y : [y_0,y_1]; a list of training traget for non-geometric method
  '''
  #use coarse graining with (2,2) kernel then contrust pore network
  CG_G,n,e,c = pix_to_graph(coarse_grain(pore_array,kernel_size=ks),kernel_size=ks);
  print(f'The generated pore network has {n} nodes')

  #loading and coarse-graining fluid velocity field
  f = np.loadtxt(vel_dir)
  f = np.reshape(f,(256,256,2))
  f = np.rot90(f)
  #coarse graining with a 2*2 kernel
  # ks = 2 #kernel size
  CG_f_0 = coarse_grain(f[:,:,0],ks)/(ks*ks)
  CG_f_1 = coarse_grain(f[:,:,1],ks)/(ks*ks)
  CG_f = [CG_f_0, CG_f_1]

  #creating X and Y
  # x = np.array([[i[0] for i in list(node_pos(CG_G).values())],[i[1] for i in list(node_pos(CG_G).values())]])
  # y_0 = np.array([CG_f_0[i] for i in list(node_pos(CG_G).values())])
  # y_1 = np.array([CG_f_1[i] for i in list(node_pos(CG_G).values())])
  # y = [y_0,y_1]
  return f, CG_G, CG_f

#Utilis for Inversion
def get_bounds(array):
  '''
  array : array of pore space
  returns a list of boundary turples
  To draw contour: obtain contours from cv2.findContours()
  - use image = cv2.drawContours(image, contours, -1, (0, 255, 0), 0)
  - plt.imshow(image)
  '''
  # image = cv2.imread(img_path)
  # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  array = array.astype(np.uint8)#adjusting
  contours, hierarchy = cv2.findContours(array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  boundary_points = []
  for i,pts in enumerate(contours):
    if pts.shape[0] == 1:
      pt = pts.squeeze().tolist()
      boundary_points.append((pt[1],pt[0]))
    else:
      for pt in pts.squeeze().tolist():
        boundary_points.append((pt[1],pt[0]))
  return boundary_points

#pore image to pore array
def gen_pore_array(img_path:str):
  an_image = PIL.Image.open(img_path)
  image_sequence = an_image.getdata()
  image_array = np.array(image_sequence)/255
  k = np.asarray(an_image)[:,:,0]/255
  #adjustment so that the pore spaces are mapped to 0 and the solid immovable parts mapped to 1
  pore_array = np.ones_like(k) - k
  return pore_array