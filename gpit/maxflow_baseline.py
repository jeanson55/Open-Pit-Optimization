#LG module is part of the gPit software  (https://github.com/GuilhermeMene/gPit)
#Copyright (c) 2023 gPit Developers
#Distributed under the terms of MIT License (https://github.com/GuilhermeMene/gPit/blob/main/LICENSE)

#Implements the Lerchs-Grossmann ultimate pit limit as an exact maximum-flow /
#minimum-cut solve. LG and Hochbaum's Pseudoflow (used in pseudoflow.py) both
#solve the same maximum-closure problem on the same precedence graph -- they
#are proven to return mathematically identical optimal solutions (Hochbaum,
#2008; Picard, 1976). This module deliberately reuses the SAME graph
#construction as Pseudoflow.UPL (same node numbering scheme via precedence.py,
#same external-arc rule via CreateExtArcs) so that any difference between an
#LG.UPL() result and a Pseudoflow.UPL() result on the same inputs is a real
#discrepancy worth investigating, not an artefact of two different graphs.

import numpy as np
import networkx as NetX
import pandas as pd

from gpit import precedence as prec
from gpit import logger as log


class  MaxFlowBaseline:

    def __init__(self, blockmodel, bmParms):

        self.resultUpl = {}

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

        #Create the Sink (same numbering convention as Pseudoflow: source=0,
        #blocks=1..n, sink=n+1)
        self.sink = np.int64(self.bmParms['nx'] * self.bmParms['ny'] * self.bmParms['nz'] + 1)

    def UPL(self, EPV_column:str, precedence='1x5x21', uplName='upl'):

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

                #Create the external arcs (identical rule to Pseudoflow.CreateExtArcs)
                self.Graph = self.CreateExtArcs(self.Graph)

                #Create internal (precedence) arcs -- identical loop structure
                #and identical prec.* template functions as Pseudoflow.UPL, so
                #the graph topology matches exactly.
                if precedence == '1x5' or precedence == '1x9':
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

                #Solve the maximum-closure problem exactly via min-cut/max-flow.
                #This is what makes it "LG": Lerchs-Grossmann and max-flow/min-cut
                #both solve maximum closure on a precedence graph and are
                #provably equivalent -- we get networkx's exact preflow-push
                #max-flow implementation to do the min-cut instead of running
                #Hochbaum's pseudoflow algorithm.
                #
                #Every edge added by CreateExtArcs / prec.* carries a 'const'
                #attribute (the arc's absolute capacity); networkx's flow
                #routines expect a 'capacity' attribute, so we mirror it in.
                for _, _, d in self.Graph.edges(data=True):
                    d['capacity'] = d['const']

                cut_value, (source_side, _sink_side) = NetX.minimum_cut(
                    self.Graph, source, self.sink,
                    capacity='capacity', flow_func=NetX.algorithms.flow.preflow_push,
                )

                #Blocks on the source side of the cut (excluding the source
                #node itself) are the blocks inside the ultimate pit -- same
                #convention as Pseudoflow.UPL's cuts.items() == [1] filter.
                InsideList = [n for n in source_side if n != source]

                for node in InsideList:
                    self.upl[np.int64(node - 1)] = 1

            except Exception as e: log.datalogger(e)

        else:
            log.datalogger("The pit has negative values only.")

        #Save the result of the UPL into dict
        self.resultUpl[uplName] = self.upl

        return self

    def CreateExtArcs(self, Graph):

        #Identical rule to Pseudoflow.CreateExtArcs: profitable blocks get an
        #arc from the source, waste blocks get an arc to the sink.
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

        #Run different pits based on revenue factors -- identical pattern to
        #Pseudoflow.UPL_nestedPits so nested-shell results can be compared
        #shell-by-shell as well as at the ultimate-pit level.
        if isinstance(Rflist, list):
            try:
                for rf in range(len(Rflist)):
                    rfname = Rflist[rf].split('_')
                    rfname = f"UPL_RF_{rfname[2]}"

                    self.RfUPLlist.append(rfname)

                    self.UPL(EPV_column=Rflist[rf], precedence=precedence, uplName=rfname)

            except Exception as e: log.datalogger(e)

        else:
            log.datalogger("The revenue factors must be a list.")
        return self
