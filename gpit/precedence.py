#Precedence module is part of the gPit software  (https://github.com/GuilhermeMene/gPit)
#Copyright (c) 2023 gPit Developers 
#Distributed under the terms of MIT License (https://github.com/GuilhermeMene/gPit/blob/main/LICENSE)

#Based on Sebastian Avalos code disributed under MIT License 
#https://code.engineering.queensu.ca/geomet-group/2020-pseudoflow-python

import networkx as NetX

def IA1x9(pos_x, pos_y, pos_z, nx, ny, Graph):

    #Create positions
    p_0 =  1 + pos_x + nx*pos_y + ny*nx*pos_z    
    p_1 =  1 + (pos_x-1) + nx*(pos_y-1) + ny*nx*(pos_z+1)
    p_2 =  1 + (pos_x) + nx*(pos_y-1) + ny*nx*(pos_z+1)
    p_3 =  1 + (pos_x+1) + nx*(pos_y-1) + ny*nx*(pos_z+1)
    p_4 =  1 + (pos_x-1) + nx*(pos_y) + ny*nx*(pos_z+1)
    p_5 =  1 + (pos_x) + nx*(pos_y) + ny*nx*(pos_z+1)
    p_6 =  1 + (pos_x+1) + nx*(pos_y) + ny*nx*(pos_z+1)
    p_7 =  1 + (pos_x-1) + nx*(pos_y+1) + ny*nx*(pos_z+1)
    p_8 =  1 + (pos_x) + nx*(pos_y+1) + ny*nx*(pos_z+1)
    p_9 =  1 + (pos_x+1) + nx*(pos_y+1) + ny*nx*(pos_z+1)
    
    #Create edge
    Graph.add_edge(p_0, p_1, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2, const=99e99, mult=1)
    Graph.add_edge(p_0, p_3, const=99e99, mult=1)
    Graph.add_edge(p_0, p_4, const=99e99, mult=1)
    Graph.add_edge(p_0, p_5, const=99e99, mult=1)
    Graph.add_edge(p_0, p_6, const=99e99, mult=1)
    Graph.add_edge(p_0, p_7, const=99e99, mult=1)
    Graph.add_edge(p_0, p_8, const=99e99, mult=1)
    Graph.add_edge(p_0, p_9, const=99e99, mult=1)
    
    return Graph

def IA1x5(pos_x, pos_y, pos_z, nx, ny, Graph):

    #Create positions
    p_0 =  1 + pos_x + nx*pos_y + ny*nx*pos_z
    p_2 =  1 + (pos_x) + nx*(pos_y-1) + ny*nx*(pos_z+1)
    p_4 =  1 + (pos_x-1) + nx*(pos_y) + ny*nx*(pos_z+1)
    p_5 =  1 + (pos_x) + nx*(pos_y) + ny*nx*(pos_z+1)
    p_6 =  1 + (pos_x+1) + nx*(pos_y) + ny*nx*(pos_z+1)
    p_8 =  1 + (pos_x) + nx*(pos_y+1) + ny*nx*(pos_z+1)

    #Create edge
    Graph.add_edge(p_0, p_2, const=99e99, mult=1)
    Graph.add_edge(p_0, p_4, const=99e99, mult=1)
    Graph.add_edge(p_0, p_5, const=99e99, mult=1)
    Graph.add_edge(p_0, p_6, const=99e99, mult=1)
    Graph.add_edge(p_0, p_8, const=99e99, mult=1)

    return Graph

def IA1x1x5(pos_x, pos_y, pos_z, nx, ny, Graph):

    #Create positions
    p_0 =  1 + pos_x + nx*pos_y + ny*nx*pos_z
    p_1_0 = 1 + pos_x + nx*pos_y + ny*nx*(pos_z+1)
    p_2_2 =  1 + (pos_x) + nx*(pos_y-1) + ny*nx*(pos_z+2)
    p_2_4 =  1 + (pos_x-1) + nx*(pos_y) + ny*nx*(pos_z+2)
    p_2_5 =  1 + (pos_x) + nx*(pos_y) + ny*nx*(pos_z+2)
    p_2_6 =  1 + (pos_x+1) + nx*(pos_y) + ny*nx*(pos_z+2)
    p_2_8 =  1 + (pos_x) + nx*(pos_y+1) + ny*nx*(pos_z+2)

    #Create edge
    Graph.add_edge(p_0, p_1_0, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_2, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_4, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_5, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_6, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_8, const=99e99, mult=1)

    return Graph

def IA1x5x13(pos_x, pos_y, pos_z, nx, ny, Graph):

    #Create positions 
    p_0 =  1 + pos_x + nx*pos_y + ny*nx*pos_z               
    p_1_8 =  1 + (pos_x) + nx*(pos_y-1) + ny*nx*(pos_z+1)     
    p_1_12 =  1 + (pos_x-1) + nx*(pos_y) + ny*nx*(pos_z+1)    
    p_1_13 =  1 + (pos_x) + nx*(pos_y) + ny*nx*(pos_z+1)       
    p_1_14 =  1 + (pos_x+1) + nx*(pos_y) + ny*nx*(pos_z+1)     
    p_1_18 =  1 + (pos_x) + nx*(pos_y+1) + ny*nx*(pos_z+1)     
    p_2_3 = 1 + (pos_x) + nx*(pos_y+2) + ny*nx*(pos_z+2)
    p_2_7 = 1 + (pos_x-1) + nx*(pos_y+1) + ny*nx*(pos_z+2)
    p_2_8 = 1 + (pos_x) + nx*(pos_y+1) + ny*nx*(pos_z+2)
    p_2_9 = 1 + (pos_x+1) + nx*(pos_y+1) + ny*nx*(pos_z+2)
    p_2_11 = 1 + (pos_x-2) + nx*(pos_y) + ny*nx*(pos_z+2)
    p_2_12 = 1 + (pos_x-1) + nx*(pos_y) + ny*nx*(pos_z+2)
    p_2_13 = 1 + (pos_x) + nx*(pos_y) + ny*nx*(pos_z+2)
    p_2_14 = 1 + (pos_x+1) + nx*(pos_y) + ny*nx*(pos_z+2)
    p_2_15 = 1 + (pos_x+2) + nx*(pos_y) + ny*nx*(pos_z+2)
    p_2_17 = 1 + (pos_x-1) + nx*(pos_y-1) + ny*nx*(pos_z+2)
    p_2_18 = 1 + (pos_x) + nx*(pos_y-1) + ny*nx*(pos_z+2)
    p_2_19 = 1 + (pos_x+1) + nx*(pos_y-1) + ny*nx*(pos_z+2)
    p_2_23 = 1 + (pos_x) + nx*(pos_y-1) + ny*nx*(pos_z+2)

    #Create edge 
    Graph.add_edge(p_0, p_1_8, const=99e99, mult=1)
    Graph.add_edge(p_0, p_1_12, const=99e99, mult=1)
    Graph.add_edge(p_0, p_1_13, const=99e99, mult=1)
    Graph.add_edge(p_0, p_1_14, const=99e99, mult=1)
    Graph.add_edge(p_0, p_1_18, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_3, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_7, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_8, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_9, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_11, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_12, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_13, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_14, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_15, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_17, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_18, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_19, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_23, const=99e99, mult=1)

    return Graph

def IA1x5x21(pos_x, pos_y, pos_z, nx, ny, Graph):

    #Create positions 
    p_0 =  1 + pos_x + nx*pos_y + ny*nx*pos_z               
    p_1_8 =  1 + (pos_x) + nx*(pos_y-1) + ny*nx*(pos_z+1)     
    p_1_12 =  1 + (pos_x-1) + nx*(pos_y) + ny*nx*(pos_z+1)    
    p_1_13 =  1 + (pos_x) + nx*(pos_y) + ny*nx*(pos_z+1)       
    p_1_14 =  1 + (pos_x+1) + nx*(pos_y) + ny*nx*(pos_z+1)     
    p_1_18 =  1 + (pos_x) + nx*(pos_y+1) + ny*nx*(pos_z+1)     
    p_2_2 = 1 + (pos_x-1) + nx*(pos_y+2) + ny*nx*(pos_z+2)
    p_2_3 = 1 + (pos_x) + nx*(pos_y+2) + ny*nx*(pos_z+2)
    p_2_4 = 1 + (pos_x+1) + nx*(pos_y+2) + ny*nx*(pos_z+2)
    p_2_6 = 1 + (pos_x-2) + nx*(pos_y+1) + ny*nx*(pos_z+2)
    p_2_7 = 1 + (pos_x-1) + nx*(pos_y+1) + ny*nx*(pos_z+2)
    p_2_8 = 1 + (pos_x) + nx*(pos_y+1) + ny*nx*(pos_z+2)
    p_2_9 = 1 + (pos_x+1) + nx*(pos_y+1) + ny*nx*(pos_z+2)
    p_2_10 = 1 + (pos_x+2) + nx*(pos_y+1) + ny*nx*(pos_z+2)
    p_2_11 = 1 + (pos_x-2) + nx*(pos_y) + ny*nx*(pos_z+2)
    p_2_12 = 1 + (pos_x-1) + nx*(pos_y) + ny*nx*(pos_z+2)
    p_2_13 = 1 + (pos_x) + nx*(pos_y) + ny*nx*(pos_z+2)
    p_2_14 = 1 + (pos_x+1) + nx*(pos_y) + ny*nx*(pos_z+2)
    p_2_15 = 1 + (pos_x+2) + nx*(pos_y) + ny*nx*(pos_z+2)
    p_2_16 = 1 + (pos_x-2) + nx*(pos_y-1) + ny*nx*(pos_z+2)
    p_2_17 = 1 + (pos_x-1) + nx*(pos_y-1) + ny*nx*(pos_z+2)
    p_2_18 = 1 + (pos_x) + nx*(pos_y-1) + ny*nx*(pos_z+2)
    p_2_19 = 1 + (pos_x+1) + nx*(pos_y-1) + ny*nx*(pos_z+2)
    p_2_20 = 1 + (pos_x+2) + nx*(pos_y-1) + ny*nx*(pos_z+2)
    p_2_22 = 1 + (pos_x-1) + nx*(pos_y-1) + ny*nx*(pos_z+2)
    p_2_23 = 1 + (pos_x) + nx*(pos_y-1) + ny*nx*(pos_z+2)
    p_2_24 = 1 + (pos_x+1) + nx*(pos_y-1) + ny*nx*(pos_z+2)

    #Create edge 
    Graph.add_edge(p_0, p_1_8, const=99e99, mult=1)
    Graph.add_edge(p_0, p_1_12, const=99e99, mult=1)
    Graph.add_edge(p_0, p_1_13, const=99e99, mult=1)
    Graph.add_edge(p_0, p_1_14, const=99e99, mult=1)
    Graph.add_edge(p_0, p_1_18, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_2, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_3, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_4, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_6, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_7, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_8, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_9, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_10, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_11, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_12, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_13, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_14, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_15, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_16, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_17, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_18, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_19, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_20, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_22, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_23, const=99e99, mult=1)
    Graph.add_edge(p_0, p_2_24, const=99e99, mult=1)

    return Graph