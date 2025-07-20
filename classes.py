# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 10:43:53 2025

@author: lorenz
"""

import numpy as np
import pickle
import json

#===================================================================================


class Graph:
    """
    Minimal class representing a directed graph using an adjacency list.
    Serves as a base class for Weighted_Graph. We omit defining attributes like nodes or
    edges and there are no methods, since we don't need it for our purposes.
    
    Attributes:
        -adj_list (list of arrays): adj_list[v] is an integer array containing the nodes w 
                               such that (v,w) forms an edge.
    """
    def __init__(self, neighbors_list):
        self.adj_list = neighbors_list



class Weighted_Graph(Graph):
    """
    Weighted_Graph is a subclass of Graph that assigns weights to the edges based on the 
    reciprocal Euclidean distance between connected nodes.

    Attributes:
        -adj_list (list of arrays): Inherited from Graph.
        -weights (list of arrays): weights[v] contains the weights of edges (v, w) 
                              for all neighbors w of node v.
    """
    def __init__(self, neighbors_list, array_coords):
        # initialize parent attributes via neighbors_list
        super().__init__(neighbors_list)
        
        #initialize list of weights which entries are computed below
        self.weights = [0]*len(neighbors_list)
        
        
        for v in range(len(neighbors_list)):
            
            # we add all weights up to compute weight of reflexive edges (v,v) later
            total = 0.0
            
            # the neighbors w of node v are adj_list[v]
            array_neighbors = self.adj_list[v]
            
            # initialize array for storing the weights of all edges between v and its neighbors
            array_tmp_weights = np.zeros(len(neighbors_list[v]), dtype=float)
            
            for w in array_neighbors:
                
                # only if v != w, we can compute reciprocal distance
                if v != w:
                    
                    dist = np.linalg.norm(array_coords[v] - array_coords[w])
                    weight = 1.0 / dist
                    
                    #computed weight is placed at the position of node w
                    array_tmp_weights[np.where(array_neighbors==w)] = weight
                    total += weight
                    
            # in the end, we can also add the weight of the reflexive edge
            array_tmp_weights[np.where(array_neighbors==v)] = -total
            
            self.weights[v] = array_tmp_weights
            
            



# ====================================================================


class CRS_matrix(Weighted_Graph):
    """
    CRS_matrix is a subclass of Weighted_Graph. The objects are matrices which are
    stored in the CRS format, i.e. the attributes are: 
                -adj_list (list of arrays): inherited from Graph
                -weights (list of arrays): inherited from Weighted_Graph
                
                -values (list of arrays): all non-zero entries of the matrix
                -col_index (list of arrays): the column indices of each non-zero entry 
                -row_ptr (array): the cumulative number of non-zero entries in each row
    
    The methods of CRS_matrix are:
                -matrix_vector_product: realizes the matrix vector product A·x 
                -save_matrix: saves objects as binary or ASCII file
                -read_matrix: recovers objects from binary or ASCII file
    """
    
    def __init__(self, neighbors_list, array_coords):
        super().__init__(neighbors_list, array_coords)

        # The way we constructed the parent/grandparent class allows us to define 
        # col_index=adj_list and values=weights. 
        self.values=self.weights
        self.col_index=self.adj_list
        
        #row_ptr consists of the cumulative sums of the lengths of the elements of weights
        self.row_ptr= np.concatenate((np.array([0], dtype=int), np.cumsum([len(array) for array in self.weights])))
       
    
    def matrix_vector_product(self, vector):
        """
        Computes the matrix-vector product A·x, where A is the current CRS_matrix object.

        Input:
            vector (np.array): array representing the vector to be multiplied.
                               Its length must match the number of columns of the matrix.

        Return:
            vec_output (np.array): The resulting vector from the matrix-vector multiplication.

        """
        
        # exemplary trouble shooting, there are other potential problems possible, like
        # vector not being an array or containing no numerical values.
        if(len(vector) != len(self.row_ptr)-1):
            print("length of vector:", len(vector), "does not match matrix dimensions:",
                  len(self.row_ptr)-1, "x",len(self.row_ptr)-1)
            
            vec_output = np.array([], dtype=float)
        else:
            
            #initialize output vector as array of zeros, we also use that our matrix is square
            vec_output=np.zeros(len(self.row_ptr)-1, dtype=float)

            for v in range(0, len(self.row_ptr)-1): 
                vec_output[v]=np.dot(self.values[v], vector[self.col_index[v]])
        
        return vec_output
    
    
    def save_matrix(self, name, mode):
        """
        Saves the CRS_matrix object to the current directory in either binary (pkl) 
        or ASCII (JSON) format.

        Input:
           name (str): The base name of the output file (without extension).
           mode (str): File format to use; must be either "pkl" or "json".

        Return:
           None
        """
        
        # minimalistic troubleshooting for wrong input
        if mode not in ["binary","ASCII"]:
            print("mode must either be binary or ascii")
        
        else:
            if mode=="binary":
                # add suffix .pkl to create "pickle" file
                filename= name + ".pkl"
                # open file in binary mode ('wb') and save current object
                with open(filename, 'wb') as file:
                    pickle.dump(self, file)
             
             
            else: #suffix == "pkl" 
                filename= name + ".json"
                #change the arrays in values and col_index to a list which is required to use json.dump 
                content = {
                        "values": [array.tolist() for array in self.values],
                        "col_index":  [array.tolist() for array in self.col_index],
                        "row_ptr": self.row_ptr.tolist()
                        }
                with open(filename, 'w') as f:
                    # indent=2 leads to better formatting in the .json file
                    json.dump(content, f, indent=2)
     
            
    @classmethod
    def read_matrix(cls, filename):        
        """
        Reads a CRS_matrix object from a file in either ASCII (JSON) or binary (PKL) format.

        Input:
            filename (str): The name of the file to read (including extension ".json" or ".pkl").

        Return:
            matrix: The reconstructed CRS_matrix object.
        """
        
        # we split filename according to . and save the substring after the last "."
        suffix = filename.split(".")[-1]
        
        #minimal trouble shooting: we only allow txt and pkl files
        if suffix not in ["json", "pkl"]:
            print("No feasible fileformat")
        
        else:
            if suffix == "json":
                with open(filename, "r") as f:
                    data = json.load(f)
                
                #define "empty" CSR_matrix object whose attributes will be asigned below
                matrix = cls.__new__(cls)
                
                #we expect the txt file to be of the form created in method save_file
                #we again create a list of arrays as it is intended for values
                #and col_index
                matrix.values = [np.array(array, dtype=float) for array in data["values"]]
                matrix.col_index = [np.array(array, dtype=float) for array in data["col_index"]]
                matrix.row_ptr = np.array(data["row_ptr"], dtype=int)
            
            else: # suffix == "pkl"      
                # pickle allows us to directly recover class object
                with open(filename, "rb") as file:
                    matrix = pickle.load(file)                   
                
            return matrix    

            