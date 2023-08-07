# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:26:57 2023

@author: every
"""

import PySimpleGUI as sg

def run_renewable_models():
    # Replace this function with your code for running renewable energy models
    print("Running Renewable Models")

def run_demand_models():
    # Replace this function with your code for running future demand models
    print("Running Future Demand Models")

def run_plant_optimizers():
    # Replace this function with your code for running plant optimizers
    print("Running Plant Optimizers")

def run_energy_management_system():
    # Replace this function with your code for running the energy management system
    print("Running Energy Management System")

# Define the layout of the left column
left_column = [
    [sg.Button('Renewable Models', key='-RENEWABLE-', size=(20, 2))],
    [sg.Button('Future Demand Models', key='-DEMAND-', size=(20, 2))],
    [sg.Button('Plant Optimizers', key='-OPTIMIZERS-', size=(20, 2))],
]

# Define the layout of the right column
right_column = [
    [sg.Button('Energy Management System', key='-EMS-', size=(20, 2))],
    # Add more buttons on the right side if needed
]

# Define the layout of the arrows
arrows = [
    [sg.Canvas(size=(30, 50), background_color='white', key='-ARROW-')],
]

# Combine the left and right columns in the main layout
layout = [
    [
        sg.Column(left_column),
        sg.Column(arrows),
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
    elif event == '-RENEWABLE-':
        run_renewable_models()
    elif event == '-DEMAND-':
        run_demand_models()
    elif event == '-OPTIMIZERS-':
        run_plant_optimizers()
    elif event == '-EMS-':
        run_energy_management_system()

window.close()
