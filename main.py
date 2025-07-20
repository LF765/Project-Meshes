# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 08:36:11 2025

@author: lorenz
"""

"""
General comments on the classes and functions defined in functions.py and classes.py:

- Performance considerations:
    Some implemented operations and data structures are not optimized for high performance,
    such as the matrix-vector product in CRS_matrix or certain for-loops that could be
    parallelized. The primary intention is to provide correct and working examples
    for the problems defined in the problem sheet. There is certainly
    room for improvement regarding high-performance computing aspects.

- User-friendliness and error handling:
    The implemented methods and structures are primarily tailored to the examples
    used in main.py. While minimal error checking is included (e.g., file format validation
    in read_matrix), most methods require careful use. In a professional context,
    significantly more robust error handling and input validation would be necessary.

- External code and AI usage:
    All classes, data structures, and methods were conceptualized and implemented by myself.
    The use of external sources (e.g., GitHub or language models) was limited to
    minor issues such as file I/O handling or specific list/array operations.
    
How to use main.py:

At the beginning of the script, you can choose to load either the small or the large dataset.
After loading the data, the tasks from Problem 1 and Problem 2 are executed.

You may run the entire file at once or go through each problem individually,
as the script is clearly divided into sections marked by problem number.
"""


import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix #to test matrix-vector product

# read in the functions from functions.py
from functions import node_to_elements, element_to_elements, node_to_nodes, find_nearest_element 
# classes from classes.py
from classes import Graph, Weighted_Graph, CRS_matrix 

#================================================================================================


# read data as array, skip first row which only consists of the number of elements
array_coords= np.loadtxt("3x3_coords_2d.txt", skiprows=1, dtype=float)
array_elements=np.loadtxt("3x3_elements_2d.txt", skiprows=1, dtype=int)



# uncommend to read large data set
#array_coords =np.loadtxt("117x34_coords_2d.txt", skiprows=1, dtype=float)
#array_elements=np.loadtxt("117x34_elements_2d.txt", skiprows=1, dtype=int)



#========================================================================================================
#=================================== Problem 1 a)====================================================
#=====================================================================================================

#========================= mapping node_to_elements ===================================================

node= int(5)
# try out function node_to_elements
elements = node_to_elements(node, array_elements)
print("The elements", elements, "contain node", node)


# R is the number of nodes (=numbers of rows of array_small_coords), C=3 for our data, since we 
# work with triangles
R,C=array_coords.shape

# create data structure nodes_to_elements which is a list of R many arrays. The v-th
# entry of this list is an array containing all neighboring elements of node
nodes_to_elements = [node_to_elements(node, array_elements) for node in range(R)]



#========================= mapping element_to_elements =================================================


element= int(5)
elements= element_to_elements(element, array_elements)
print("The neighbors of element", element, "are the elements", elements)


# R is the number of elements (=numbers of rows of array_elements), C=3 for our data, since we 
# work with triangles
R,C=array_elements.shape

# create data structure elements_to_elements which is a list of R many arrays. The v-th
# entry of this list is an array containing all neighboring elements of the element v 
# including v itself.
elements_to_elements = [node_to_elements(node, array_elements) for node in range(R)]



#=========================== mapping node_to_nodes =====================================================



neighbors= node_to_nodes(node, array_elements)
print("The neighboring nodes of node", node, "are", neighbors)


# create data structure nodes_to_nodes which is a list of R many arrays. The v-th
# entry of this list is an array containing all adjacent nodes of node v
# including v itself.
R,C=array_coords.shape
nodes_to_nodes = [node_to_nodes(node, array_elements) for node in range(R)]



#========================================================================================================
#=================================== Problem 1 b)====================================================
#=================================================================================================


### Test the function find_nearest_element for a few random points in [-0.5,1.5]^2

n = 5
points = np.random.uniform(low=-0.5, high=1.5, size=(n, 2))


for i in range(n):
    closest_ele= find_nearest_element(points[i], array_coords, array_elements)
    print(closest_ele, "is the closest element to ", points[i])
    

            
            

    
 
#========================================================================================================
#=================================== Problem 1 c)====================================================
#=====================================================================================================

"""
To store a mixed discretization with both triangles and quadrilaterals, 
the data structure representing elements must be adapted to support a variable number
of vertices per element. Instead of using a fixed-size array (with shape Nx3),
one should for example use a list of arrays. This would also require to change the functions
in functions.py as they rely on the currently used data structure.

"""



#========================================================================================================
#=================================== Problem 2 a)====================================================
#=====================================================================================================


#We can initialize "Graph" and "Weighted_Graph" objects using the nodes_to_nodes data structure
#from Problem 1a). Weighted_Graph also requires the nodal coordinates
g = Graph(nodes_to_nodes)
g_weighted=Weighted_Graph(nodes_to_nodes, array_coords)



#========================================================================================================
#=================================== Problem 2 b) ====================================================
#=====================================================================================================

# generate the adjacency matrix of our weighted graph using class CRS_matrix which
# requires the nodes_to_nodes relation from task 1a) and the nodal coordinates

matrix_adjacency=CRS_matrix(nodes_to_nodes, array_coords)


#==============  c) Test Matrix-Vector multiplication for correctness ======================================

# define control matrix using already build csr matrix class
# recall that values is a list of arrays, where the length of the list
# is the number of rows of our matrix
nrow= len(matrix_adjacency.values)
ncol=nrow

matrix_control = csr_matrix((np.concatenate(matrix_adjacency.values),
                             np.concatenate(matrix_adjacency.col_index), 
                             matrix_adjacency.row_ptr), shape=(nrow, nrow))


# random normally distributed entries in vector
vector = np.random.randn(ncol)



# use matrix-vector multiplication for control matrix
output_control = matrix_control @ vector

# compute output with class method matrix_vector_product
output_vec=matrix_adjacency.matrix_vector_product(vector)

#compare the results
print("Difference between control and class method:", np.linalg.norm(output_control-output_vec))



#============================ c) Write/read binary and ascii files =====================================================


#save matrix_adjacency as binary file
name="crs_matrix"
matrix_adjacency.save_matrix(name, mode="binary")

#save matrix_adjacency as ASCII file
name="crs_matrix"
matrix_adjacency.save_matrix(name, mode="ASCII")



#read object as binary file
crs_matrix = CRS_matrix.read_matrix("crs_matrix.pkl")


#read object as ascii file
crs_matrix = CRS_matrix.read_matrix("crs_matrix.json")



#========================================================================================================
#=================================== Problem 2 d) ====================================================
#=====================================================================================================

"""
In the following, we plot the mask of our adjacency matrix (matrix_adjacency) using a scatter plot.
While there are packages such as NetworkX that allow visualizing the structure of a (weighted) graph,
this seemed not meaningful for large inputs like the 117x34 dataset, as the resulting plot would be too
dense to convey useful information.
"""

# extract values from class and concatenate to array
values = np.concatenate(matrix_adjacency.values)

# extract all column indices from the non-zero entries and concatenate to array
col_indices = np.concatenate(matrix_adjacency.col_index)


#in order to plot matrix pattern, we need an array of all row indices of all non-zero entries
row_ptr= matrix_adjacency.row_ptr


list_row_indices = [np.full(row_ptr[v+1]-row_ptr[v],v, dtype=int) for v in range(0, len(row_ptr)-1)]

# convert to array
row_indices=np.concatenate(list_row_indices)

num_rows=len(row_ptr)-1
num_cols=num_rows

# visualize all non-zero entries of matrix_adjacency
plt.figure(figsize=(10, 10))

scatter = plt.scatter(col_indices, row_indices,
                      s=20,               # size of points
                      edgecolors='k')     # black boundary


plt.title("Non--zero entries of CSR-matrix")
plt.xlabel("columnindex")
plt.ylabel("rowindex")

# revert y axis position 0 on top
plt.gca().invert_yaxis()

# change axis size to matrix size
plt.xlim(-0.5, num_cols - 0.5)
plt.ylim(num_rows - 0.5, -0.5)

# add grid
plt.grid(True, which='both', color='grey', linestyle=':', linewidth=0.5)

plt.show()
