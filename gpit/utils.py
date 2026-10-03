#Utils module is part of the gPit software  (https://github.com/GuilhermeMene/gPit)
#Copyright (c) 2023 gPit Developers 
#Distributed under the terms of MIT License (https://github.com/GuilhermeMene/gPit/blob/main/LICENSE)


import pandas as pd
import numpy as np 

import os

from gpit import logger as log
from gpit import LOGPATH as path

import pyvista as pv 
import meshio

def mergeNP(blockmodel, PitNames):

    #Function to merge differents nested pits into one column 
    #Prepare the list of pits for correct interations 
    PitNames.sort(reverse=True)

    if isinstance(PitNames, list): 
        if isinstance(blockmodel, pd.DataFrame):
            #Create a numpy array of zeros 
            mergedPit = np.zeros(len(blockmodel))
            try:
                for pn in range(0, len(PitNames)):
                    for idx, row in blockmodel.iterrows():   
                        if row[PitNames[pn]] == 1:
                            mergedPit[idx] = pn+1
                blockmodel['Pits'] = mergedPit
            except Exception as e: log.datalogger(e)            
        else: 
            log.datalogger("The Blockmodel must be a pandas dataframe. ")
    else: 
        log.datalogger("The Pit Names must be a list of nested pits. ")

    return blockmodel

def calcNPStats(blockmodel, PitNames, epvCol, blockTonnes):

    #Function to calculate the statistics from nested pits
    #This function return a table with NPV, ore tonnes, and waste tonnes of nested pits 

    #Prepare the list of pits for correct interations 
    PitNames.sort()

    if isinstance(PitNames, list) and isinstance(epvCol, list): 

        if len(PitNames) == len(epvCol): 
            #Create the variables 
            orevolume = np.zeros(len(PitNames))
            wastevolume = np.zeros(len(PitNames))
            npv = np.zeros(len(PitNames))

            if isinstance(blockmodel, pd.DataFrame):
                try:
                    for pn in range(0, len(PitNames)):

                        pitbm = blockmodel[blockmodel[PitNames[pn]] == 1]

                        orevolume[pn] = (pitbm[epvCol[pn]][pitbm[epvCol[pn]] > 0].count() * blockTonnes)
                        wastevolume[pn] = (pitbm[epvCol[pn]][pitbm[epvCol[pn]] < 0].count() * blockTonnes)
                        npv[pn] = (pitbm[epvCol[pn]].sum())
                except Exception as e: log.datalogger(e)

            else: 
                log.datalogger("The blockmodel must be a pandas dataframe. ")

        else: 
            log.datalogger("The length of Pitnames must be equal to the length of the epvCol")

    else: 
        log.datalogger("The PitNames and epvCol must be a list. ")


    #Create a pandas dataframe 
    d = {
        'Pit': PitNames, 
        'OreVolume': orevolume,
        'WasteVolume': wastevolume,
        'NPV': npv
    }
    stats = pd.DataFrame(data=d)

    return stats

def calcStat(blockmodel, PitCol:str, epvCol:str, blockTonnes):

    #Function to calculate a single column statistics
    if isinstance(blockmodel, pd.DataFrame):
        if isinstance(epvCol, str):

            pitbm = blockmodel[blockmodel[PitCol] == 1]

            orevolume = (pitbm[epvCol][pitbm[epvCol] > 0].count() * blockTonnes)
            wastevolume = (pitbm[epvCol][pitbm[epvCol] < 0].count() * blockTonnes)
            npv = pitbm[epvCol].sum()

        else: 
            log.datalogger("The epvCol must be a string with the column name to calculate the final pit mesh. ")
    else: 
        log.datalogger("the blockmodel must be a pandas dataframe. ")


    return orevolume, wastevolume, npv


def getPitCoords(blockmodel, pitCol:str, xCol='X', yCol='Y', zCol='Z'):

    #Function to calculate the mesh from ultimate pit limit 
    if isinstance(blockmodel, pd.DataFrame):
        if isinstance(pitCol, str):

            try:
                mesh_v_index = []

                count = 0   #DEBUG ONLY

                #Get the first slice of the blockmodel 
                xy = blockmodel[blockmodel[zCol] == max(blockmodel[zCol])]
                xy = xy[[xCol, yCol]]

                for i, row in xy.iterrows():

                    column = blockmodel[(blockmodel[xCol] == row[xCol]) & (blockmodel[yCol] == row[yCol])]

                    if column[pitCol].eq(1).any().any():
                        col = column[column[pitCol] == 1] #Filter the outside pit data
                        index = int(col[col[zCol] == min(col[zCol])].index.values)
                    else:
                        index = int(column[column[zCol] == max(column[zCol])].index.values)
                        
                    count += 1 #DEBUG ONLY
                    #Save the index 
                    mesh_v_index.append(index)

                #Get the coordinates from list 
                df = blockmodel.iloc[mesh_v_index]

                df = df[[xCol, yCol, zCol]]

            except Exception as e: log.datalogger(e)

        else: 
            log.datalogger("The pitCol must be a string with the colume name to calculate the stats. ")
    else: 
        log.datalogger("the blockmodel must be a pandas dataframe. ")
        
    return df

def createMesh(blockmodel, pitCol:str, filename:str, xCol='X', yCol='Y', zCol='Z'):

    #Function to calculate the mesh from ultimate pit limit 
    if isinstance(blockmodel, pd.DataFrame):
        if isinstance(pitCol, str):
            try: 

                coords = getPitCoords(blockmodel, pitCol, xCol='X', yCol='Y', zCol='Z')
                points = np.column_stack([coords[xCol], coords[yCol], coords[zCol]])
                cloud = pv.PolyData(points)
                mesh = cloud.delaunay_2d()

                #Save the mesh file 
                filename = os.path.join(path, filename)
                meshio.Mesh(points, {"triangle": mesh.regular_faces}).write(filename)

            except Exception as e: log.datalogger(e)
        else:
            log.datalogger("The pitCol must be a string with name of the Ultimate Pit Limit.")
    else: 
        log.datalogger("the blockmodel must be a pandas dataframe with xyz coordinates, and pit data. ")













