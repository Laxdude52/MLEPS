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

dataModels = dict()
 
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

def createList(files):
    tmpDataList = []
    for i in range(len(files)):
        tmpData = pd.read_csv(files[i])
        tmpDataList.append(tmpData)
    return tmpDataList



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