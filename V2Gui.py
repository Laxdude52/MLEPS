# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 13:39:12 2023

@author: School Account
"""

import PySimpleGUI as sg

def dataManageWindow():
    print("Start Data Management Window")
    window.hide()
    
    left_column = [
        [sg.Button('Load Data', key='-Load Data-', size=(20,3))],
        [sg.Button('Create Data', key='-Create Data-', size=(20,3))],
        ]
    right_column = [
        [sg.Text("Loaded Datasets")],
        [sg.Listbox(['test1', 'test2', 'test3', 'test4', 'test5'], size=(20,7), key='-Dataframe')],
        [sg.Button('Manage Selected Dataframe', key='-Manage Selected Data-', size=(20,2))],
        ]
    
    dataManage_layout = [
        [
            sg.Column(left_column),
            sg.VerticalSeparator(),
            sg.Column(right_column),
            ]
        ]
    
    dataManage_window = sg.Window('Data Management', dataManage_layout)
    
    while True:
        event_dataManage, values_dataManage = dataManage_window.read()
        
        if event_dataManage == sg.WIN_CLOSED:
            dataManage_window.close()
            window.un_hide()
            break
        

# Define the layout of the left column
left_column = [
    [sg.Button('Manage Data', key='-Data-', size=(20, 3))],
    [sg.Button('Manage Models', key='-Models-', size=(20, 3))],
]

# Define the layout of the right column
right_column = [
    [sg.Button('Energy Management System', key='-EMS-', size=(25, 8))],
    # Add more buttons on the right side if needed
]

# Combine the left and right columns in the main layout
layout = [
    [
        sg.Column(left_column),
        sg.VerticalSeparator(),
        sg.Column(right_column),
    ]
]

# Create the window
window = sg.Window('Energy Management GUI', layout)

# Event loop
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break
    elif event == '-Data-':
        print("Data")
        dataManageWindow()
    elif event == '-Models-':
        print("Models")
        #modelManageWindow()
    elif event == '-EMS-':
        print("EMS")
        #EMSWindow()

window.close()