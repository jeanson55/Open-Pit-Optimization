#Pseudoflow module is part of the gPit software  (https://github.com/GuilhermeMene/gPit)
#Copyright (c) 2023 gPit Developers 
#Distributed under the terms of MIT License (https://github.com/GuilhermeMene/gPit/blob/main/LICENSE)

#Based on Sebastian Avalos code disributed under MIT License 
#https://code.engineering.queensu.ca/geomet-group/2020-pseudoflow-python

import numpy as np
import networkx as NetX
import pseudoflow as pf
import pandas as pd

from gpit import precedence as prec
from gpit import logger as log

class Pseudoflow:

    def __init__(self, blockmodel, bmParms):

        self.resultUpl ={}

        try: 
            #Create the inputs 
            if isinstance(blockmodel, pd.DataFrame):
                self.blockmodel = blockmodel
            else: 
                log.datalogger("The blockmodel must be a pandas Dataframe")

            if len(bmParms) == 10: 
                self.bmParms = bmParms
            else: 
                log.datalogger("The blockmodel Parameters must contain 10 parameters - Set the blockmodel parameters first")

        except Exception as e: log.datalogger(e)

        #Create the Sink 
        self.sink = np.int64(self.bmParms['nx'] * self.bmParms['ny'] * self.bmParms['nz'] + 1)

    def UPL(self, EPV_column:str, precedence = '1x5x21', uplName = 'upl'):

                #Set the EPV variable 
        self.EPV = np.asarray(self.blockmodel[EPV_column])

        #Create a numpy array with zeros 
        self.upl = np.zeros(len(self.EPV))

        #Check positive or negative values
        if any(val < 0 for val in self.EPV):

            source = 0

            try:      

                #Set the Graph 
                self.Graph = NetX.DiGraph()  

                #Create the external arcs 
                self.Graph = self.CreateExtArcs(self.Graph)

                if precedence == '1x5' or precedence == '1x9':
                    #Create Internal arcs 
                    for ind_z in range(self.bmParms['nz'] - 1):
                        pos_z = self.bmParms['nz'] - ind_z - 2
                        for pos_y in range(ind_z + 1, self.bmParms['ny'] - ind_z - 1):
                            for pos_x in range(ind_z + 1, self.bmParms['nx'] - ind_z - 1):

                                if precedence == '1x5':
                                    self.Graph = prec.IA1x5(pos_x, pos_y, pos_z, self.bmParms['nx'], self.bmParms['ny'], self.Graph)
                                elif precedence == '1x9':
                                    self.Graph = prec.IA1x9(pos_x, pos_y, pos_z, self.bmParms['nx'], self.bmParms['ny'], self.Graph)
                                else:
                                    log.datalogger("The precedence must be '5x1', '9x1', '1x5x13', '1x5x21' or '1x1x5' ")
                                    break

                else: 
                    #Create Internal arcs 
                    for ind_z in range(self.bmParms['nz'] - 2):
                        pos_z = self.bmParms['nz'] - ind_z - 3
                        for pos_y in range(ind_z + 2, self.bmParms['ny'] - ind_z - 2):
                            for pos_x in range(ind_z + 2, self.bmParms['nx'] - ind_z - 2):

                                if precedence == '1x5x13':
                                    self.Graph = prec.IA1x5x13(pos_x, pos_y, pos_z, self.bmParms['nx'], self.bmParms['ny'], self.Graph)
                                elif precedence == '1x5x21':
                                    self.Graph = prec.IA1x5x21(pos_x, pos_y, pos_z, self.bmParms['nx'], self.bmParms['ny'], self.Graph)
                                elif precedence == '1x1x5':
                                    self.Graph = prec.IA1x1x5(pos_x, pos_y, pos_z, self.bmParms['nx'], self.bmParms['ny'], self.Graph)
                                else: 
                                    log.datalogger("The precedence must be '5x1', '9x1', '1x5x13', '1x5x21' or '1x1x5' ")
                                    break

                #Solving the minimum cut problem via pf.hpf solver
                RngLambda = [0]
                breakpoints, cuts, info = pf.hpf(self.Graph, source, self.sink, const_cap="const", mult_cap="mult", lambdaRange=RngLambda, roundNegativeCapacity=False)

                #Get the UPL inside values 
                B = {x:y for x, y in cuts.items() if y == [1] and x!=0}
                InsideList = list(B.keys())

                for indUPL in range(len(InsideList)):
                    # Set blocks inside UPL as one
                    self.upl[np.int64(InsideList[indUPL] - 1)] = 1  

            except Exception as e: log.datalogger(e)

        else:
            log.datalogger("The pit has negative values only.")

        #Save the result of the UPL into dict 
        self.resultUpl[uplName] = self.upl

        return self 

    def CreateExtArcs(self, Graph):

        for t_z in range(self.bmParms['nz']):
            pos_z = self.bmParms['nz'] - t_z - 1
            for t_y in range(t_z, self.bmParms['ny'] - t_z):
                for t_x in range(t_z, self.bmParms['nx'] - t_z):
                    p_i = 1 + t_x + self.bmParms['nx']*t_y + self.bmParms['ny']*self.bmParms['nx']*pos_z 
                    Capacity = np.absolute(np.around(self.EPV[p_i-1], decimals=2))
                    if self.EPV[p_i-1] < 0: 
                        Graph.add_edge(p_i, self.sink, const=Capacity, mult=-1)
                    else:
                        Graph.add_edge(0, p_i, const=Capacity, mult=1)

        return Graph
    
    def UPL_nestedPits(self, Rflist, precedence='1x5x21'):

        self.RfUPLlist = []

        #Run different pits based on revenue factors 
        if isinstance(Rflist, list):
            try: 
                for rf in range(len(Rflist)):
                    #Get the name of the upl 
                    rfname = Rflist[rf].split('_')
                    rfname = f"UPL_RF_{rfname[2]}"

                    self.RfUPLlist.append(rfname)

                    self.UPL(EPV_column=Rflist[rf], precedence=precedence, uplName=rfname)

            except Exception as e: log.datalogger(e)

        else:
            log.datalogger("The revenue factors must be a list.")
        return self

