# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:52:19 2023

@author: School Account
"""

import machineLearningTool as aim
import data as ud
import pandas as pd
import copy
import pickle 
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

def createData(goal, modelType, group):
    parameters = ud.dataModels[goal][modelType][group]
    
    dataList = parameters['dataList']
    dataPack = parameters['dataPack']
    columns = dataPack['columns']
    
    if modelType.upper() == 'SIMPLE':
        simpData = aim.data(dataList)
        simpData.baseFilter(columns['all'], dataPack['aboveVal'], dataPack['belowVal'], dataPack['naDecision'])
        simpData.split(dataPack['testSize'], dataPack['validSize'], columns['y'], columns['X'], shuffle=True)
        
        ud.updateInitData(simpData, goal, 'Simple', group)
    elif modelType.upper() == 'FUTURE':
        data = aim.data(dataList)
        data.baseFilter(columns['all'], dataPack['aboveVal'], dataPack['belowVal'], dataPack['naDecision'])
        multiTime = False
        if len(dataPack['timeCols']) > 1:
            multiTime = True
        data.aggrigate(dataPack['aggTime'], dataPack['timeCols'], multiTime)
        data.lagFuture(columns['y'], dataPack['lagTime'], dataPack['notLagTime'], dataPack['numPastSteps'], dataPack['numFutureSteps'])
        data.split(dataPack['testSize'], dataPack['validSize'], columns['y'], data.input_cols)
        
        ud.updateInitData(data, goal, 'Future', group)
    else:
        warnings.warn("INCORRECT MODEL TYPE ENTERED (Not simple/future")

def createModel(goal, modelType, group):
    parameters = ud.dataModels[goal][modelType][group]
    
    model = parameters['model']
    initData = parameters['initData']
    modelPack = parameters['modelPack']
    structure = modelPack['structure']
    
    #Create Model
    if structure.upper() == 'REGULAR':
        simpModel = aim.model(modelPack['model'], modelPack['modelType'], modelPack['param_dist'], initData)
        simpModel.train(1, modelPack['inerators'])
    elif structure.upper() == 'ENSEMBLE':
        simpModel = aim.ensemble(modelPack['modelList'], modelPack['newModelDataframe'], initData)
        simpModel.create_Base()
        simpModel.lastLayer(modelPack['lastList'])
    elif structure.upper() == 'BREAKOUT':
        simpModel = aim.breakout(modelPack['clusterSearch'], modelPack['cluster'], modelPack['clusterName'], modelPack['cluster_params'], modelPack['run_params'], initData)
        simpModel.startCluster()
        simpModel.createData(prepTrain=True)
        simpModel.trainSecond(modelPack['secondLayerModel'])
    else:
        warnings.warn("INCORRECT MODEL STRUCTURE")

    if modelType.upper() == 'SIMPLE':
        ud.updateModel(simpModel, goal, 'Simple', group)
    elif modelType.upper() == 'FUTURE':
        ud.updateModel(simpModel, goal, 'Future', group)
    else:
        warnings.warn("INCORRECT MODEL TYPE (not simple/future)")
        
        
#Add code to write models to computer below