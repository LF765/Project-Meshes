# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 09:27:13 2025

@author: lorenz
"""

# please install packages if not installed yet
import numpy as np


#==============================================================================

"""
The following helper functions are used to construct the data structures required for Problem 1a).
They are small and mostly self-explanatory in their purpose and implementation.
"""

## mapping that finds all elements containing a given node
def node_to_elements(node, elements):
    
    # extract dimensions, for our data C=3
    R,C=elements.shape
    

    list_elements=[]
    
    # iterate over columns to find all elements of which node is part of
    for k in range(C):  
        
        # where finds the index l such that elements[l,k]=node                            
        list_elements.append(np.where(elements[:,k]==node)[0])
        

    # return sorted and concatenated array
    return np.sort(np.concatenate(list_elements))



#===============================================================================
## mapping that finds, for a given element, all neighboring elements,
# i.e. all elements that share a node with the given element

def element_to_elements(element, elements):

    # extract all nodes which define element
    array_nodes=elements[element]
    
    list_neighbors=[]

    # for each node that defines input element, find all other elements that contain node
    for node in array_nodes:

        list_neighbors.append(node_to_elements(node, elements))
    
    # concatenate list to array, remove duplicates and sort the elements
    array_neigbors=np.sort(np.unique(np.concatenate(list_neighbors)))
    
    return array_neigbors


#================================================================================

## The following mapping finds, for a given node, all neighbouring nodes and returns them
#  including node itself

def node_to_nodes(node, elements):

    # find all elements which contain node
    adjacent_elements=node_to_elements(node, elements)
    
    neighboring_nodes_list = []

    # each element in the for loop below contains node, thus node is also part of the output
    for element in adjacent_elements:
        neighboring_nodes_list.append(elements[element])

    # concatenate combines all the arrays from neighboring_nodes_list to one single array 
    array_neighboring_nodes = np.concatenate(neighboring_nodes_list)
    
    return np.unique(array_neighboring_nodes)
    



#================================ Part b) =================================================


def find_nearest_element(point, array_coords, array_elements):
    
    """
    Finds the nearest mesh element to a given 2D point.

    In the function we first identify the node in the mesh that is closest to the input point,
    then we consider only those mesh elements that contain this node. Among those, we select
    the element whose polygonal shape is closest (in Euclidean sense) to the input point.

    Parameters:
    ----------
    point : np.array of shape (2,)
        The 2D coordinate for which the nearest element is to be found.

    array_coords : np.ndarray of shape (N, 2)
        Array containing the coordinates of all N mesh nodes.

    array_elements : Each entry `array_elements[v]` is an array containing the indices of the nodes 
        that form the v-th mesh element.
        
    Return:
    -------
    closest_element : int
        Index of the element in `array_elements` that is closest to the input point.
    """
    # find the node which is the closest to point
    differences = array_coords - point
    dists = np.linalg.norm(differences, axis=1)
    nearest_node=np.argmin(dists)
    
    #create array of all neighboring elements of nearest_node
    array_neighboring_elements= node_to_elements(nearest_node, array_elements)
    
    
    closest_dist=np.inf
    
    for element in array_neighboring_elements:
        
        #extract coordinates of current element
        array_ele_coords= array_coords[array_elements[element]] 
        
        
        # compute distance from coordinate to current element
        new_dist= dist_to_polygon(point, array_ele_coords)
        
        # if new_dist < closest_dist, we have found a closer element
        if new_dist < closest_dist:
            closest_dist=new_dist
            closest_element=element
        
        
            
    return closest_element

 




#====================== dist_to_polygon ======================================

def dist_to_polygon(coord, array_ele_coords):
    """
    Computes the distance from a given 2D point to a polygon defined by its vertex coordinates.

    The function uses the ray-casting algorithm (https://en.wikipedia.org/wiki/Point_in_polygon)
    to check whether the point lies inside the polygon. If the point is inside, the distance 
    is zero. If the point is outside, it returns the minimal Euclidean distance to the 
    polygon's boundary (i.e., to any of its edges).

    This function is used as a helper in find_nearest_element.

    Input:
    ----------
    coord : np.array of shape (2,)
        2D coordinate (x, y) from which the distance to the polygon is computed.

    array_ele_coords : np.array of shape (N, 2)
        Array containing the coordinates of the N vertices of the polygonal element.
        The vertices should be ordered to represent a closed polygon (no need to repeat the first point;
        this is handled internally).

    Return:
    -------
    distance : float
        Distance from `coord` to the polygon. Returns 0.0 if the point is inside,
        otherwise the minimal Euclidean distance to the polygon's boundary.
    """
    
    polygon = np.concatenate((array_ele_coords, array_ele_coords[0][None, :]))
    x, y = coord
    count = 0
    # number of edges
    n = len(polygon) - 1  

    #apply ray-casting to test whether coord is in polygon
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[i + 1]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                count += 1
    
    # if count is odd, coord is inside polygon
    inside = (count % 2 == 1)
    if inside:
        distance = 0.0 
    
    else: 
        distance = np.inf
        #compute minimal distance to all edges of polygon
        for i in range(n):
            
            a = polygon[i]
            b = polygon[i + 1]
            ab = b - a
            ac = coord - a
            t = np.dot(ac, ab) / np.dot(ab, ab)  #t = argmin( s \in R : || coord - (a+s*ab) ||_2)
            
            # all points on the edge are of the form a + s*ab for 0<=s<=1
            t_edge = np.clip(t, 0, 1)
            
            nearest = a + t_edge * ab
            dist = np.linalg.norm(coord - nearest)
            if dist < distance:
                distance = dist

    return distance
              