# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:20:59 2023

@author: every
"""  

import PySimpleGUI as sg
    
def renewablesWindow():
    print("opening renewables window")
    
    window.hide()
    
    solar_feature = [
        [sg.Text('Select model target')],
        [sg.DropDown(values=['Kw', 'POAI', 'GHI', 'TmpF'], key='-Solar Target Drop-', readonly=True, enable_events=True)],
        [sg.Text('Real Time Models:')],
        [sg.Listbox(['Option 1', 'Option 2', 'Option 3', 'Option 4'], size=(20, 3), key='-Solar Real Drop-', enable_events=True, disabled=True)],
        [sg.Text('Future Models:')],
        [sg.Button('View', key='Solar Real View', size=(10,1), disabled=True)],
        [sg.Listbox(['Option 1', 'Option 2', 'Option 3', 'Option 4'], size=(20, 3), key='-Solar Future Drop-', enable_events=True, disabled=True)],
        [sg.Button('View', key='Solar Future View', size=(10,1), disabled=True)],
        ]

    wind_feature = [
        [sg.Text('Select model target')],
        [sg.DropDown(values=['wind1', 'wind2', 'wind3'], key='-Wind Target Drop-', readonly=True, enable_events=True)],
        [sg.Text('Real Time Models:')],
        [sg.Listbox(['Option 1', 'Option 2', 'Option 3', 'Option 4'], size=(20, 3), key='-Wind Real Drop-', enable_events=True, disabled=True)],
        [sg.Text('Future Models:')],
        [sg.Button('View', key='Wind Real View', size=(10,1), disabled=True)],
        [sg.Listbox(['Option 1', 'Option 2', 'Option 3', 'Option 4'], size=(20, 3), key='-Wind Future Drop-', enable_events=True, disabled=True)],
        [sg.Button('View', key='Wind Future View', size=(10,1), disabled=True)],
        ]
    
    renewables_layout = [
        [
            sg.Column(
                [[sg.Frame('Solar', solar_feature)]],
                 element_justification='left'),
            sg.VerticalSeparator(),
            sg.Column(
                [[sg.Frame('Wind', wind_feature)]],
                element_justification='right'),
            ],
        [sg.Column([[sg.Button('Create New', key='newModel', size=(46,1))]], element_justification='center')],
        ]
    
    renewables_window = sg.Window('Renewables', renewables_layout)
    
    while True:
        event_renewables, values_renewables = renewables_window.read()

        if event_renewables == sg.WIN_CLOSED:
            # Close the Renewables window and show the original window again
            renewables_window.close()
            window.un_hide()
            break
        elif event_renewables == '-Solar Target Drop-':
            renewables_window['-Solar Real Drop-'].update(disabled=False)
            renewables_window['Solar Real View'].update(disabled=False)
            renewables_window['-Solar Future Drop-'].update(disabled=False)
            renewables_window['Solar Future View'].update(disabled=False)
        elif event_renewables == '-Wind Target Drop-':
            renewables_window['-Wind Real Drop-'].update(disabled=False)
            renewables_window['Wind Real View'].update(disabled=False)
            renewables_window['-Wind Future Drop-'].update(disabled=False)
            renewables_window['Wind Future View'].update(disabled=False)

def create_models(mType):
    
    
    
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
    [sg.Button('Renewable Models', key='-RENEWABLE-', size=(20, 3))],
    [sg.Button('Future Demand Models', key='-DEMAND-', size=(20, 3))],
    [sg.Button('Plant Optimizers', key='-OPTIMIZERS-', size=(20, 3))],
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
    elif event == '-RENEWABLE-':
        renewablesWindow()
    elif event == '-DEMAND-':
        run_demand_models()
    elif event == '-OPTIMIZERS-':
        run_plant_optimizers()
    elif event == '-EMS-':
        run_energy_management_system()

window.close()
