# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:02:41 2023

@author: School Account
"""
'''
Define the data structure layout:
    
    
dataModels -> goal -> modelType (simple/future) -> group -> {
                                    Model
                                    Initialized Data
                                    Data List (Used to create initialized data)
                                    Model Pack (All model parameters used for training)
                                    Data Pack (Data parameters)
                                    save (boolean to save or not)
                                    }
'''

import pandas as pd
import warnings 
import copy

dataModels = dict()

def createElecData():
    historicalElectricityData = copy.deepcopy(dataModels['Demand']['Simple']['default']['initData'].filtered)
    return historicalElectricityData
 
def addGoal(newName):
    dataModels.update({newName:dict()})
    dataModels[newName].update({"Simple":dict()})
    dataModels[newName].update({"Future":dict()})

def addGroup(newName, goal, modelType):
    newGroup = dict()
    newGroup.update({"model":'NA'})
    newGroup.update({"initData":'NA'})
    newGroup.update({"dataList":'NA'})
    newGroup.update({"modelPack":'NA'}) 
    newGroup.update({"dataPack":'NA'})
    newGroup.update({"save":True})
    if modelType.upper() == 'SIMPLE':
        dataModels[goal]['Simple'].update({newName:newGroup})
    elif modelType.upper() == 'FUTURE':
        dataModels[goal]['Future'].update({newName:newGroup})
    else:
        warnings.warn("INCORRECT MODEL TYPE ENTERED - SIMPLE OR FUTURE ONLY")

def updateModel(model, goal, modelType, group):
    dataModels[goal][modelType][group].update({"model":model})
def updateInitData(initData, goal, modelType, group):
    dataModels[goal][modelType][group].update({"initData":initData})
def updateDataList(dataList, goal, modelType, group):
    dataModels[goal][modelType][group].update({"dataList":dataList})
def updateModelPack(modelPack, goal, modelType, group):
    dataModels[goal][modelType][group].update({"modelPack":modelPack})
def updateDataPack(dataPack, goal, modelType, group):
    dataModels[goal][modelType][group].update({"dataPack":dataPack})
def updateSave(save, goal, modelType, group):
    dataModels[goal][modelType][group].update({"save":save})

def predict(X, goal, modelType, group): 
    val = dataModels[goal][modelType][group]['model'].model.predict(X)
    return val

def livePredict(timeStamp, goal, modelType, group):
    X_train = dataModels[goal][modelType][group]['initData'].X_train 
    X = X_train.iloc[timeStamp]
    val = dataModels[goal][modelType][group]['model'].model.predict(X)
    return val 

def createList(files):
    tmpDataList = []
    for i in range(len(files)):
        tmpData = pd.read_csv(files[i])
        tmpDataList.append(tmpData)
    return tmpDataList

plantInformation = dict()

def newPlant(name, informationDict):
    plantInformation.update({name:informationDict})
def updatePlantEfficency(name, data):
    plantInformation[name]['efficencyData'] = data

def defPredict(goal, modelType, group):
    X = dataModels[goal][modelType][group]['initData'].X_test
    y = dataModels[goal][modelType][group]['initData'].y_test
    results, predValues = dataModels[goal][modelType][group]["model"].modPredict(X, y, returnResults=True, returnPred=True)
    print(results)
    print(predValues)
    return results, predValues


def mostEfficent(plantName, plantLevel):
    setData = plantInformation[plantName]['efficencyData'].iloc[plantLevel]
    setLevel = setData[setData.columns[0]]
    minLevel = setLevel - plantInformation[plantName]['MaxRamp']
    maxLevel = setLevel + plantInformation[plantName]['MaxRamp']
    
    if(maxLevel > plantInformation[plantName]['Max']):
        maxLevel = plantInformation[plantName]['Max']
    if(minLevel < 0):
        minLevel = 0
    
    minHeatRate = -1
    for i in range(minLevel, maxLevel):
        if(minHeatRate == -1):
            minHeatRate = plantInformation[plantName]['efficencyData'].iloc[i]
            minHeatRate = minHeatRate[minHeatRate.columns[0]]
        else:
            tmpHeatRate = plantInformation[plantName]['efficencyData'].iloc[i]
            tmpHeatRate = tmpHeatRate[tmpHeatRate.columns[0]]
            if(tmpHeatRate < minHeatRate):
                minHeatRate = tmpHeatRate
    return minHeatRate

#Old Code Below:
'''
dataLists = dict()

def createList(files, storedName):
    #files is a list
    tmpDataList = dict()
    for i in range(len(files)):
        tmpData = pd.read_csv(files[i])
        tmpDataList.update({i:tmpData})
    dataLists.update({storedName: tmpDataList})
    
testFiles = ["BS_2016.csv", "BS_2017.csv", "BS_2018.csv"]
createList(testFiles, "Test Solar 2016-2018")
    

#Save all Datasets hee
r2016 = pd.read_csv("BS_2016.csv")
r2017 = pd.read_csv("BS_2017.csv")
r2018 = pd.read_csv("BS_2018.csv")
r2019 = pd.read_csv("BS_2019.csv")
r2020 = pd.read_csv("BS_2020.csv")
r2021 = pd.read_csv("BS_2021.csv")
r2022 = pd.read_csv("BS_2022.csv")

solarDataList = [r2016, r2017, r2018, r2019, r2020, r2021, r2022]

windDataList = 'NA'
energyDemandList = 'NA'
coalList = 'NA'

hydroList1 = 'NA'
hydroList2 = 'NA'

combList1 = 'NA'
combList2 = 'NA'
combList3 = 'NA'
combList4 = 'NA'
combList5 = 'NA'
combList6 = 'NA'
combList7 = 'NA'
'''