# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:07:49 2023

@author: every
"""
'''
step 1 (Future):
    pred elec demand
    pred solar output
    set select to most efficent level
    then next set focusing more on meeting demand
    repeat focusing more on meeting demand and less on efficency
step 2 (Live):
    get current elec demand
    get current plant status
    get current solar status
    if below, move most efficent plants up until demand is met
    if above, move msot efficent plants down until demand is met
    

'''

import data as ud


defaultStepOrder = [["Primemover 1", 'Steam 1'], ["Primeover 2", "Turbine 1"], ["Primeover 3", "Steam 2"], ['Primeover 1', 'Turbine 2']]
futureLevel = [[0,0], [0,0]]
stepIteration = 0

def steppingAlgorithm(liveElecDemand, timeStamp, currentPlantLevels):
    #Future:
        #Calculate future electricity demand, account for solar
        predElecDemand = ud.livePredict(timeStamp, "Demand", "Future", 'default')
        predSolarOutput = ud.livePredict(timeStamp, "Solar", "Future", 'default')
        predEDemand = predElecDemand-predSolarOutput
        
        #Get most efficent levels for efficency plants
        efficencyPlant1 = defaultStepOrder[stepIteration][0]
        efficencyPlant2 = defaultStepOrder[stepIteration][1]
        
        plantLevel1 = ud.mostEfficent(efficencyPlant1, currentPlantLevels[efficencyPlant1])
        plantLevel2 = ud.mostEfficent(efficencyPlant2, currentPlantLevels[efficencyPlant2])
        
        
        