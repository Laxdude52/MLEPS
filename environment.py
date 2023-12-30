# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:26:38 2023

@author: every
"""

'''
Necessary Features

Get:
    Future Model Setting:
        1. plant levels
            a. Max ramp up/down
            b. other associated data
        2. elec demand pred
        3. solar output pred
    Current Logic Setting:
        1. plant levels
            a. Max ramp up/down
            b. other associated data
        2. real demand:
            real elec demand - real solar output (future) - other unit output (ex. battery, water)
 
Do:
    Future:
        Set future plant levels
    Logic:
        Set current plant levels

'''


import data as ud
STARTTIME = 


#Get current plant levels + max ramp up/down
def getAllPlantInformation():
    return ud.plantInformation

def getSpecPlantInformation(plant):
    return ud.plantInformation[plant]

#Get solar pred 