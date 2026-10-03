#Blockmodel module is part of the gPit software  (https://github.com/GuilhermeMene/gPit)
#Copyright (c) 2023 gPit Developers 
#Distributed under the terms of MIT License (https://github.com/GuilhermeMene/gPit/blob/main/LICENSE)

import numpy as np
import pandas as pd
from gpit import logger as log

class BlockModel:

    def __init__(self, bmpath:str):

        self.MinCost = 0
        self.ProcCost = 0

        #Set the BM parameters 
        self.EPVparms = {}

        #Read the blockmodel
        try: 
            self.blockmodel = pd.read_csv(bmpath, delimiter=',')
        except:
            try: 
                self.blockmodel = pd.read_csv(bmpath, delimiter=';')
            except Exception as e: log.datalogger(e)

        log.datalogger("The blockmodel has been loaded.")

    def setBMParms(self, nx:int, ny:int, nz:int, xorg:float, yorg:float, zorg:float, xs:int, ys:int, zs:int):
        bmparms = {
            "nx": 120,
            "ny": 120,
            "nz": 26,

            "xorg": 0,
            "yorg": 0,
            "zorg": 0,

            "xs": 1,
            "ys": 1,
            "zs": 1,

            "volume": 1,
        }

        return self   

    def setEPVParms(self, metal:str, unit:str, mprice:float, sel_cost:float, recovery:float, dilution:float):        

        #Save the values into dict 
        self.EPVparms[metal] = {
            'unit': unit,
            'mprice': mprice ,
            'sel_cost': sel_cost ,
            'recovery': recovery, 
            'dilution' : dilution
        }  

    def importEPVParms(self, EPVpath:str):

        #Set the function for import the a csv file with the economic parameters 
        #Read the epv paramters 
        try: 
            epv_df = pd.read_csv(EPVpath, delimiter=',')
        except:
            try: 
                epv_df = pd.read_csv(EPVpath, delimiter=';')
            except Exception as e: log.datalogger(e)

        if len(epv_df) > 1:
            #Get parameters from each metal in table 
            for idx, row in epv_df.iterrows():
                self.setEPVParms(metal=row['Metal'], unit=row['Unit'], mprice=row['Mprice'], sel_cost=row['SelCost'], 
                                    recovery=row['Recovery'], dilution=row['Dilution'])
        else: 
            #Set only one metal epv parameter 
            self.setEPVParms(metal=epv_df['Metal'], unit=epv_df['Unit'], mprice=epv_df['Mprice'], sel_cost=epv_df['SelCost'], 
                                recovery=epv_df['Recovery'], dilution=epv_df['Dilution'])

        return self

    def setCosts(self, mincost:float, proccost:float):

        #Check mine and proc costs 
        if self.MinCost == 0 and self.ProcCost == 0:
            self.MinCost = mincost
            self.ProcCost = proccost
        else: 
            pass

        return self      
    
    def calculateCutOff(self, metal:str, unit='percent'):

        #Check the unit of the calculation 
        if unit == 'percent':
            try: 
                self.CutOffGrade = ((self.MinCost + (self.ProcCost*(1 + self.EPVparms[metal]['dilution']))) / 
                                        (((self.EPVparms[metal]['mprice'] - self.EPVparms[metal]['sel_cost'])*22.046)*self.EPVparms[metal]['recovery']))
            except: 
                log.datalogger("Set the economic parameters first")

        elif unit == 'ozt':
            try:
                self.CutOffGrade = (((self.MinCost + (self.ProcCost*(1 + self.EPVparms[metal]['dilution']))) / 
                                    ((self.EPVparms[metal]['mprice'] - self.EPVparms[metal]['sel_cost'])*self.EPVparms[metal]['recovery']))*31.1)
            except:
                log.datalogger("Set the economic parameters first")

        else: 
            log.datalogger("Set the correct unit for the Cut-Off grade calculation")

        return self

    def calcEPV(self, OreGrade, Unit, Density, ColName:str, Rf=1):

        mrecvalue = 0 
        block_value = []

        #Check and calculate the density of each block 
        if isinstance(Density, str):
            self.tonnes = [self.bmparms['volume'] * self.blockmodel[Density] for self.blockmodel[Density] in self.blockmodel[Density]]
        if isinstance(Density, float):
            bmlen = [0] * len(self.blockmodel) 
            self.tonnes = [self.bmparms['volume'] * Density for bmlen in bmlen]

        for ind in self.blockmodel.index:

            if isinstance(OreGrade, str):
                ore_grade = float(self.blockmodel[OreGrade][ind])
            else:
                ore_grade = float(self.blockmodel[OreGrade[0]][ind])

            if ore_grade < self.CutOffGrade:
                block_value.append(round((-self.MinCost * self.tonnes[ind]), 2))
            else: 
                #Check if one or more metals 
                #Only one metal passed as string 
                if isinstance(OreGrade, str) and isinstance(Unit, str):
                    if Unit == 'ozt':
                        #Block value for process calculation
                        process_value = (((self.blockmodel[OreGrade][ind] * self.EPVparms[OreGrade]['recovery'] * self.tonnes[ind] *  
                                            (self.EPVparms[OreGrade]['mprice'] - self.EPVparms[OreGrade]['sel_cost'])) * Rf) / 31.1)

                        block_value.append(process_value - ((self.MinCost + self.ProcCost) * self.tonnes[ind]))
                    if Unit == 'percent':
                        #Block value for process calculation
                        process_value = ((self.blockmodel[OreGrade][ind]/100) * self.EPVparms[OreGrade]['recovery'] * self.tonnes[ind] *  
                                            ((self.EPVparms[OreGrade]['mprice'] - self.EPVparms[OreGrade]['sel_cost']) * 2204) * Rf)
                        
                        block_value.append(round((process_value - ((self.MinCost + self.ProcCost) * self.tonnes[ind])), 2))

                #More than one metal passed by list 
                elif isinstance(OreGrade, list) and isinstance(Unit, list):
                    for m in range(0, len(OreGrade)):
                        try:
                            if Unit[m] == 'ozt':
                                #Calculate the Troy ounce value 
                                mrecvalue += (((self.tonnes[ind] * self.blockmodel[OreGrade[m]][ind] * self.EPVparms[OreGrade[m]]['recovery'] * 
                                                (self.EPVparms[OreGrade[m]]['mprice'] - self.EPVparms[OreGrade[m]]['sel_cost'])) * Rf) / 31.1)

                            elif Unit[m] == 'percent':
                                #Calculate the Percent value
                                mrecvalue += (((self.tonnes[ind] * (self.blockmodel[OreGrade[m]][ind])/100) * self.EPVparms[OreGrade[m]]['recovery'] * 
                                                ((self.EPVparms[OreGrade[m]]['mprice'] - self.EPVparms[OreGrade[m]]['sel_cost']) * 2204)) * Rf)
                        except:
                            log.datalogger("The Unit must be 'ozt' or 'percent'", ind)

                    #Block value for process calculation
                    block_value.append(round((mrecvalue - (self.tonnes[ind] * (self.MinCost + self.ProcCost))), 2))

                else: 
                    log.datalogger("The OreGrade and Unit must be passed.")

        #Create the EPV into blockmodel dataframe 
        self.blockmodel[ColName] = block_value

        return self
    
    def calcEPVRf(self, OreGrade, Unit, Density, Rfmin:float, Rfmax:float, Rfstep:float):

        #Calculate different revenue factors for nested pits 
        RfList = []

        self.RfColList = []

        #Check the values 
        try: 
            i = Rfmin
            RfList.append(Rfmin)
            while i < Rfmax:
                i += Rfstep
                RfList.append(round(i, 2))

            for i in range(0, len(RfList)): 
                colName = f"EPV_RF_{RfList[i]}"
                self.RfColList.append(colName)
                self.calcEPV(OreGrade, Unit, Density, ColName=colName, Rf=RfList[i])

        except Exception as e: log.datalogger(e)

        return self 
