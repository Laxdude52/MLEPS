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

'''
Use for E.W. Brown
1 Solar
3 hydroelectric
1 Coal
7 Natural Gas

Models:
    1 model each except for hydroelectric, enter input manually
    
electricity demand model
coal and natural gas optimizers:
    max/min ramp + efficency rating
    1st, 2nd + and = ramp most efficent places
    
Controller:
    1 get the electricity demand
    
    Future controller:
        use unsupervised learning
        have a total efficency score of the model
        inputs: Current demand, predicted demand and optimizer results
        Model returns plant levels for next time period
        Want to increase efficency score, use as reward
        if outside 5% of actual demand heavy penalty
        use logic to get the electricity production equal to real in final step

'''

def createDefualtCoalModels():
    #Prepare the BTU data
    testFileBtu = 'eWBrown.csv'
    testBtu = pd.read_csv(testFileBtu)
    BtuDf = pd.DataFrame(testBtu)
    BtuDf.transpose(inplace=True)
    BtuDf.columns = BtuDf.iloc[0]
    BtuDf.drop(BtuDf.index[0], inplace=True)
    
    #Prepare the production data
    

def createDefaultSolar():
    ud.addGoal("Solar")
    ud.addGroup("default", "Solar", "Future")
    ud.addGroup("default", "Solar", "Simple")
    
    testFiles = ["BS_2016.csv", "BS_2017.csv", "BS_2018.csv"]
    testList = ud.createList(testFiles)
    
    ud.updateDataList(testList, "Solar", "Future", 'default')
    ud.updateDataList(testList, "Solar", "Simple", 'default')
     
    #Prepare data pack
    dataPack = dict()
    
    columns = dict()
    columns.update({'all': ['kW', 'Timestamp', 'POAI', 'TmpF']})
    columns.update({'y':'kW'})
    columns.update({'X': ['Timestamp', 'POAI', 'TmpF']})
    dataPack.update({"columns":columns})
    
    dataPack.update({"aboveVal": 'na'})
    dataPack.update({"belowVal": 0})
    dataPack.update({"naDecision":'mean'})
    dataPack.update({"testSize":0.3})
    dataPack.update({"validSize":0.1})
    dataPack.update({"timeCols":['Hour', 'Month']})
    dataPack.update({"aggTime":'H'})
    dataPack.update({'laggedVars':['kW', 'POAI', 'TmpF']})
    dataPack.update({'notLaggedVars':['Hour', 'Month']})
    dataPack.update({'numPastSteps':5})
    dataPack.update({'numFutureSteps':1})
    
    ud.updateDataPack(dataPack, "Solar", "Future", "default")
    
    print("Data pack initialized, now creating initialized data")
    cm.createData('Solar', "Future", "default")
    
    #Prepare model pack
    modelPack = dict()
    
    modelDNN = cm.createDefaultDNN("Solar", "Future", "default")
    modelPack.update({"model":modelDNN})
    modelPack.update({"structure":"regular"})
    modelPack.update({"modelType":"DNN"})
    modelPack.update({"param_dist":'NA'})
    modelPack.update({"param_dist":'NA'})
    modelPack.update({"iterators":100})
    
    ud.updateModelPack(modelPack, "Solar", "Future", "default")
    
    #Train Model
    print("Model Training")
    cm.createModel('Solar', 'Future', 'default')
    
    #Plot Loss
    print("Plot loss")
    ud.dataModels['Solar']['Future']['default']['model'].plot_loss()
    
def createDefaultElectricityDemand():
    ud.addGoal("Demand")
    ud.addGroup("default", "Demand", "Future")
    ud.addGroup("default", "Demand", "Simple")
    
    testFiles = ["elecDemandFinal.csv"]
    testList = ud.createList(testFiles)
    
    ud.updateDataList(testList, "Demand", "Future", 'default')
    ud.updateDataList(testList, "Demand", "Simple", 'default')
     
    #Prepare data pack
    dataPack = dict()
    
    columns = dict()
    columns.update({'all': ['DrawkW', 'tmin', 'snwd', 'tmax', 'snow', 'wt03', 'HolidayPresent', 'prcp', 'Timestamp']})
    columns.update({'y':'DrawkW'})
    columns.update({'X': ['tmin', 'snwd', 'tmax', 'snow', 'wt03', 'HolidayPresent', 'prcp', 'Timestamp']})
    dataPack.update({"columns":columns})
    
    dataPack.update({"aboveVal": 'na'})
    dataPack.update({"belowVal": -100})
    dataPack.update({"naDecision":'mean'})
    dataPack.update({"testSize":0.3})
    dataPack.update({"validSize":0.1})
    dataPack.update({"timeCols":['Hour', 'Month']})
    dataPack.update({"aggTime":'H'})
    dataPack.update({'laggedVars':['DrawkW', 'tmin', 'snow', 'wt03', 'prcp']})
    dataPack.update({'notLaggedVars':['Hour', 'Month', 'snwd', 'tmax', 'HolidayPresent']})
    dataPack.update({'numPastSteps':5}) 
    dataPack.update({'numFutureSteps':1})
    
    ud.updateDataPack(dataPack, "Demand", "Future", "default")
    
    print("Data pack initialized, now creating initialized data")
    cm.createData('Demand', "Future", "default")
    
    #Prepare model pack
    modelPack = dict()
    
    modelDNN = cm.createDefaultDNN("Demand", "Future", "default")
    modelPack.update({"model":modelDNN})
    modelPack.update({"structure":"regular"})
    modelPack.update({"modelType":"DNN"})
    modelPack.update({"param_dist":'NA'})
    modelPack.update({"param_dist":'NA'})
    modelPack.update({"iterators":300})
    
    ud.updateModelPack(modelPack, "Demand", "Future", "default")
    
    #Train Model
    print("Model Training")
    cm.createModel('Demand', 'Future', 'default')
    
