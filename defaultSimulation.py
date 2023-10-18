# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:27:16 2023

@author: School Account
"""

import machineLearningTool as aim
import data as ud
import createModels as cm
import pandas as pd
import copy
import pickle 
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

def createDefaultSolar():
    ud.addGoal("Solar")
    ud.addGroup("default", "Solar", "Future")
    ud.addGroup("default", "Solar", "Simple")
    
    testFiles = ["BS_2016.csv", "BS_2017.csv", "BS_2018.csv"]
    testList = createList(testFiles)
    
    ud.updateDataList(testList, "Solar", "Future", 'default')
    ud.updateDataList(testList, "Solar", "Simple", 'default')
     
    #Prepare data pack
    dataPack = dict()
    
    columns = dict()
    columns.update({'all': ['kW', 'timeStamp' 'POAI', 'TmpF']})
    columns.update({'y':'kW'})
    columns.update({'X': ['timeStamp', 'POAI', 'TmpF']})
    dataPack.update({"columns":columns)}
    
    dataPack.update({"aboveVal": 100})
    dataPack.update({"belowVal": 0})
    dataPack.update({"naDecision":'mean'})
    dataPack.update({"testSize":0.3})
    dataPack.update({"testSize":0.1})
    dataPack.update({"timeCols":['Hour', 'Month']})
    dataPack.update({'lagTime':})
    
    