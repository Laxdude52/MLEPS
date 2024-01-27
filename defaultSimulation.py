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
import pygam as pg

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
        
        Plant Info:
            #Assume primemover 1 is coal plant 
            
            Coal1: 457 mW
                10 hours to start, 41.2 mW/hr ramp speed
            Coal: 2-4% of max output/minute
            
            Turbines:
                full load in 30 minutes
                4 produce 110 mW
                2 produce 164 mW
                1 produce 127 mW  
                100
            
'''
plantList = list()
plantList.append("Primemover 1")
plantList.append("Primemover 2")
plantList.append("Primemover 3")
plantList.append("Steam 1")
plantList.append("Steam 2")
plantList.append("Turbine 1")
plantList.append("Turbine 2") 

def initDefaultSimulation():
    createDefualtPlantModels()
    defaultPlantInformation()
    createDefaultSolar()
    createDefaultElectricityDemand()
    
    #Verify all models working
    #Generate their grid (power output vs. heat rate)
    for i in range(len(plantList)):
        maxOut = ud.plantInformation[plantList[i]]['Max']
        tmpData = list()
        for j in range(maxOut):
            predEfficency = ud.predict(j, plantList[i], 'Simple', 'default')
            print("EfficencyVal: " + str(predEfficency))
            tmpData.append(predEfficency)
        tmpData = pd.DataFrame(tmpData)
        tmpData['output'] = tmpData.index
        print(tmpData)
        
    ud.defPredict("Solar", "Future", 'default')
    ud.defPredict("Demand", "Future", 'default')
    

def defaultSteppingPlantControl():
        #This algorithm operated by setting each plant to most efficent level within max ramp speed 
        elecData = ud.createElecData()
        
    

def defaultPlantInformation():
    prime1Info = dict()
    prime1Info.update({"Max":625})
    prime1Info.update({"MaxRamp":62.5})
    prime1Info.update({"CurrentLevel":0})
    ud.newPlant("Primemover 1", prime1Info)
    
    prime2Info = dict()
    prime2Info.update({"Max":19})
    prime2Info.update({"MaxRamp":19})
    prime2Info.update({"CurrentLevel":0})
    ud.newPlant("Primemover 2", prime2Info)
    
    prime3Info = dict()
    prime3Info.update({"Max":243})
    prime3Info.update({"MaxRamp":150})
    prime3Info.update({"CurrentLevel":0})
    ud.newPlant("Primemover 3", prime3Info)
    
    steam1Info = dict()
    steam1Info.update({"Max":625})
    steam1Info.update({"MaxRamp":100})
    steam1Info.update({"CurrentLevel":0})
    ud.newPlant("Steam 1", steam1Info)
    
    steam2Info = dict()
    steam2Info.update({"Max":3})
    steam2Info.update({"MaxRamp":3})
    steam2Info.update({"CurrentLevel":0})
    ud.newPlant("Steam 2", steam2Info)
    
    turbine1Info = dict()
    turbine1Info.update({"Max":11})
    turbine1Info.update({"MaxRamp":11})
    turbine1Info.update({"CurrentLevel":0})
    ud.newPlant("Turbine 1", turbine1Info)
    
    turbine2Info = dict()
    turbine2Info.update({"Max":25})
    turbine2Info.update({"MaxRamp":25})
    turbine2Info.update({"CurrentLevel":0})
    ud.newPlant("Turbine 2", turbine2Info)
    
    #Generate their grid (power output vs. heat rate)
    for i in range(len(plantList)):
        maxOut = ud.plantInformation[plantList[i]]['Max']
        efficencyData = list()
        efficencyPerOut = list()
        for j in range(maxOut):
            predEfficency = ud.predict(j, plantList[i], 'Simple', 'default')
            print("EfficencyVal: " + str(predEfficency))
            efficencyData.append(predEfficency)
            if(j==0):
                efficencyPerOutVal = predEfficency*2
            else:
                efficencyPerOutVal = predEfficency/j
            efficencyPerOut.append(efficencyPerOutVal)
        efficencyData = pd.DataFrame(efficencyData)
        efficencyPerOut = pd.DataFrame(efficencyPerOut)
        efficencyData = pd.concat([efficencyData,efficencyPerOut], axis=1)
        efficencyData['output'] = efficencyData.index
        ud.updatePlantEfficency(plantList[i], efficencyData)
         

def createDefualtPlantModels():
    for i in range(len(plantList)):
        ud.addGoal(plantList[i])
        ud.addGroup("default", plantList[i], 'Simple')
    
    #Prepare the BTU data
    prime1File = ["primemover1.csv"]
    prime2File = ["primemover2.csv"]
    prime3File = ["primemover3.csv"]
    #prime4File = ["primemover4.csv"]
    steam1File = ["steam1.csv"]
    steam2File = ["steam2.csv"]
    turbine1File = ["turbine1.csv"]
    turbine2File = ["turbine2.csv"]
    prime1List = ud.createList(prime1File)
    prime2List = ud.createList(prime2File)
    prime3List = ud.createList(prime3File)
    #prime4List = ud.createList(prime4File)
    steam1List = ud.createList(steam1File)
    steam2List = ud.createList(steam2File)
    turbine1List = ud.createList(turbine1File)
    turbine2List = ud.createList(turbine2File)
    
    ud.updateDataList(prime1List, "Primemover 1", 'Simple', 'default')
    ud.updateDataList(prime2List, "Primemover 2", 'Simple', 'default')
    ud.updateDataList(prime3List, "Primemover 3", 'Simple', 'default')
    #ud.updateDataList(prime4List, "Primemover 4", 'Simple', 'default')
    ud.updateDataList(steam1List, "Steam 1", 'Simple', 'default')
    ud.updateDataList(steam2List, "Steam 2", 'Simple', 'default')
    ud.updateDataList(turbine1List, "Turbine 1", 'Simple', 'default')
    ud.updateDataList(turbine2List, "Turbine 2", 'Simple', 'default')
    
    dataPack = dict()
    
    columns = dict()
    columns.update({'all':['megawatthours', 'Fuel MMBtus']})
    columns.update({'y':'Fuel MMBtus'})
    columns.update({'X':['megawatthours']})
    dataPack.update({"columns":columns})
    
    dataPack.update({"aboveVal": 'na'})
    dataPack.update({"belowVal": 0})
    dataPack.update({"naDecision":'zero'})
    dataPack.update({"testSize":0.3})
    dataPack.update({"validSize":0.1})
    dataPack.update({"scaleVal":'na'})

    modelPack = dict()
    modelPack.update({"modelType":'GAM'})
    modelPack.update({"param_dist":'NA'})
    modelPack.update({"iterators":1000})
    modelPack.update({"structure":"regular"})
    
    for i in range(len(plantList)):
        print(i)
        ud.updateDataPack(copy.deepcopy(dataPack), plantList[i], "Simple", "default")
        cm.createData(plantList[i], "Simple", "default")
        tmpModelPack = copy.deepcopy(modelPack)
        #gamModel = cm.createDefaultGAM(30, 1000, plantList[i], "Simple", "default")
        gam = pg.LinearGAM(n_splines=30, verbose=True, max_iter=1000)
        tmpModelPack.update({"model":gam})
        ud.updateModelPack(tmpModelPack, plantList[i], "Simple", "default")
        
        print("Train model") 
        cm.createModel(plantList[i], "Simple", "default")

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
    dataPack.update({"scaleVal":"na"})
    
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
    modelPack.update({"iterators":30})
    
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
    tmpScaleVal = [0,400000]
    dataPack.update({'scaleVal':tmpScaleVal})
    
    ud.updateDataPack(dataPack, "Demand", "Future", "default")
    ud.updateDataPack(dataPack, "Demand", "Simple", "default")
    
    print("Data pack initialized, now creating initialized data")
    cm.createData('Demand', "Future", "default")
    cm.createData('Demand', "Simple", "default")

    #Prepare model pack
    modelPack = dict()
    
    modelDNN = cm.createDefaultDNN("Demand", "Future", "default")
    modelPack.update({"model":modelDNN})
    modelPack.update({"structure":"regular"})
    modelPack.update({"modelType":"DNN"})
    modelPack.update({"param_dist":'NA'})
    modelPack.update({"param_dist":'NA'})
    modelPack.update({"iterators":15})
    
    ud.updateModelPack(modelPack, "Demand", "Future", "default")
    
    #Train Model
    print("Model Training")
    cm.createModel('Demand', 'Future', 'default')
    
