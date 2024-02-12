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

import warnings
import tensorflow as tf
import numpy as np
import data as ud
import defaultSimulation as sim
import logSimulation as los
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
MAXTIME = 44316
MAX_DEMAND = 3795
MAX_RAMP_RATE = 350
MAX_SOLAR = 10
solarModelInfo = ['default', 'Solar']
currentTime = 0

# Define environment parameters
state_dim = 31  # Define the dimension of the state space
action_dim = 100572000000 # Define the dimension of the action space 11524500000000000
learning_rate = 0.001
gamma = 0.99  # Discount factor 

# Get current plant levels + max ramp up/down
def getAllPlantInformation():
    return ud.plantInformation

#Get Plant information
def getSpecPlantInformation(plant):
    return ud.plantInformation[plant]


# Get solar pred
def getSolarPred(currentTime):
    # print(currentTime)
    pred = ud.livePredict(currentTime, 'Solar', 'Future', 'default', numPast=2)
    if pred > MAX_SOLAR:
        pred = MAX_SOLAR
    return pred

#Get current solar output for live simulation
def getCurrentSolar(currentTime):
    try:
        data = ud.dataModels['Solar']['Simple']['default']['initData'].y_train.iloc[[currentTime]]
    except:
        data = ud.dataModels['Solar']['Simple']['default']['initData'].y_test.iloc[[currentTime-len(ud.dataModels['Solar']['Simple']['default']['initData'].y_train)]]
    if data > MAX_SOLAR:
        data = MAX_SOLAR
    return int(data)


# Get predicted electricity demand
def fgetElecDemand(currentTime):
    # if(currentTime <= 5):
    demand = ud.livePredict(currentTime, 'Demand', 'Future', 'default', numPast=5)
    if demand > MAX_DEMAND:
        return MAX_DEMAND
    else:
        return demand

# Get real electricity demand
def rgetElecDemand(currentTime):
    #Consider two datasets
    try:
        data = ud.dataModels['Demand']['Simple']['default']['initData'].y_train.iloc[[currentTime]]
    except:
        data = ud.dataModels['Demand']['Simple']['default']['initData'].y_test.iloc[[currentTime-len(ud.dataModels['Demand']['Simple']['default']['initData'].y_train)]]
    if data > MAX_DEMAND:
        data = MAX_DEMAND
    return data

#Get total current set plant production
def getCurrentProduction(currentTime):
    tmpCurProd = 0
    for plant in sim.plantList:
        tmpPlantInfo = getSpecPlantInformation(plant)
        tmpCurProd = tmpCurProd + tmpPlantInfo["CurrentLevel"]
    tmpCurProd = int(tmpCurProd+int(getCurrentSolar(currentTime)))
    print(tmpCurProd)
    return tmpCurProd

#Reset the simulation
def resetSimulation():
    for plant in sim.plantList:
        ud.setPlantLevel(plant, ud.plantInformation[plant]["Min"])

#Get the most efficent plant output level
#Due to data constraints, the highest available output level is always the most efficent, so it only has access to half of the max ramp
def getMostEfficent(rampDir, plant):
    mostEfficentVal = 90000000
    mostEfficentSpeed = 90000000
    i = 0
    data = plant["efficencyData"].iloc[:, 1]
    # print("CurLevel" + str(plant["CurrentLevel"]))
    currentLevel = plant["CurrentLevel"]
    # print(plant)
    # print("CurLevel" + str(currentLevel))
    maxOut = plant["Max"] - 1
    if (maxOut < currentLevel + int(plant["MaxRamp"]/2)):
        upperBound = maxOut
    else:
        upperBound = int(currentLevel + (plant["MaxRamp"] / 2))
    if ((plant['Min'] + 1) > currentLevel - int(plant["MaxRamp"]/2)):
        lowerBound = plant['Min'] + 1
    else:
        lowerBound = currentLevel - int(plant["MaxRamp"]/2)
    if (rampDir == 1):
        # print("Cur: " + str(currentLevel) + "\nUppB: " + str(upperBound))
        for i in range(currentLevel, upperBound):
            if (data[i] < mostEfficentVal):
                mostEfficentVal = data[i]
                mostEfficentSpeed = i
        # print("E level" + str(mostEfficentSpeed))
    elif (rampDir == -1):
        # print("Down")
        # print("CurLevel: (Loop) " + str(currentLevel))
        # print("lowBound: (Loop) " + str(lowerBound))
        if (int(currentLevel - plant['MaxRamp'] / 2) > lowerBound):
            currentLevel = int(currentLevel - (plant['MaxRamp'] / 2))
        for i in range(lowerBound, currentLevel):
            if (data[i] < mostEfficentVal):
                mostEfficentVal = data[lowerBound]
                mostEfficentSpeed = lowerBound
    else:
        # print(rampDir)
        mostEfficentSpeed = currentLevel
        # raise Exception("No dir")
    if (mostEfficentSpeed >= plant["Max"]):
        mostEfficentSpeed = plant["Max"] - 1
    elif (mostEfficentSpeed <= plant["Min"]):
        mostEfficentSpeed = plant["Min"] + 1
    # print("Plant new level: " + str(mostEfficentSpeed) + " Plant max: " + str(plant['Max']))
    heatRate = plant["efficencyData"].iloc[:, 0][mostEfficentSpeed]
    return mostEfficentSpeed, heatRate, mostEfficentVal


def futureAlgorithm(currentTime):
    #Reinforcement model here
    print()

def currentAlgorithm(currentTime):
    tmpRealDemand = int(rgetElecDemand(currentTime))
    # print("Real elec demand: " + str(tmpRealDemand))
    tmpRealSolar = int(getCurrentSolar(currentTime))

    for plant in sim.plantList:
        tmpPlantInfo = getSpecPlantInformation(plant)
        tmpCurProd = getCurrentProduction(currentTime)
        tmpDem = tmpRealDemand - tmpCurProd
        # print(str(plant) + " Max Ramp: " + str(tmpPlantInfo["MaxRamp"]) + "  Current Level: " + str(tmpPlantInfo["CurrentLevel"]))
        tmpGap = tmpDem - tmpCurProd
        print("Gap: ", tmpGap)
        # print("Left Ramp" + str(tmpPlantInfo["LeftRamp"]))
        if (abs(tmpGap) < tmpPlantInfo["LeftRamp"]):
            newLevel = tmpPlantInfo["CurrentLevel"] + tmpGap
        else:
            if (tmpGap > 0):
                newLevel = tmpPlantInfo['CurrentLevel'] + tmpPlantInfo['LeftRamp']
            elif (tmpGap < 0):
                newLevel = tmpPlantInfo['CurrentLevel'] - tmpPlantInfo['LeftRamp']
            else:
                newLevel = tmpPlantInfo['CurrentLevel']
        if (newLevel >= tmpPlantInfo["Max"]):
            newLevel = tmpPlantInfo["Max"] - 1
        elif (newLevel <= tmpPlantInfo["Min"]):
            newLevel = tmpPlantInfo["Min"] + 1
        ud.setPlantLevel(plant, newLevel)
        ud.resetPlantRampLeft(plant)
        # print("Live level: " + str(getSpecPlantInformation(plant)["CurrentLevel"]))
        try:
            heat = tmpPlantInfo["efficencyData"].iloc[:, 0][newLevel]
        except Exception as e:
            print(e)
            print(newLevel)
            print(tmpPlantInfo['Max'])
            print(plant)
            Exception(e)
        heatPerOut = tmpPlantInfo["efficencyData"].iloc[:, 1][newLevel]
        los.logCurStep(plant, newLevel, heat, heatPerOut)
    los.logCurFutStep(currentTime, tmpRealDemand, tmpRealSolar)


'''
xVals = []
yVals = []
fig = plt.figure()

def liveGraph(xVals, yVals):
    ax1 = fig.add_subplot(1,1,1)
    xVals.append(currentTime)
    yVals.append(los.logFutDict["totalProd"][currentTime])
    xVals = xVals[-30:]
    yVals = yVals[-30:]
    ax1.clear()
    ax1.plot(xVals, yVals)  
    
    plt.title("Production over time (Future)")

'''

# sim.initDefaultSimulation()
sim.initDefaultSimulation(load=False, save=False)


# Define the neural network model
class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.dense3 = tf.keras.layers.Dense(256, activation='relu')
        self.dense4 = tf.keras.layers.Dense(action_dim, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)

# Initialize the model
policy_network = PolicyNetwork()
optimizer = tf.keras.optimizers.Adam(learning_rate)

'''
def preprocess_state(state):
    # Assuming state is a dictionary with keys 'plant_info' containing information about current production levels for each plant
    
    # Extract current levels and maximum ramp rates for all power plants
    current_levels = [plant['currentLevel'] for plant in ud.plantInformation]
    max_ramp_rates = [plant['MaxRamp'] for plant in ud.plantInformation]
    
    # Normalize the current levels and maximum ramp rates
    normalized_current_levels = normalize_data(current_levels)
    normalized_max_ramp_rates = normalize_data(max_ramp_rates)
    
    # Combine the normalized data into a single array
    processed_state = np.concatenate((normalized_current_levels, normalized_max_ramp_rates))
    
    return processed_state

def normalize_data(data):
    # Normalize data to have zero mean and unit variance
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / (std + 1e-8)  # Add a small value to avoid division by zero
    
    return normalized_data
'''

def preprocess_state(state):
    # Normalize each component of the state separately
    
    # Normalize current time (assuming it's within a specific range)
    normalized_time = state[0] / MAXTIME
    
    # Normalize predicted demand (assuming it's within a specific range)
    normalized_demand = state[1] / MAX_DEMAND
    
    # Normalize solar data (assuming it's within a specific range)
    normalized_solar = state[2] / MAX_SOLAR
    
    # Normalize information about each power plant
    normalized_plant_info = []
    for i in range(3, len(state), 4):
        current_level = state[i]  # Current level
        max_ramp_rate = state[i + 1]  # Maximum ramp rate
        min_level = state[i + 2]  # Minimum level
        max_level = state[i + 3]  # Maximum level
        
        # Normalize current level
        normalized_current_level = (current_level - min_level) / (max_level - min_level)
        
        # Normalize maximum ramp rate
        normalized_ramp_rate = max_ramp_rate / MAX_RAMP_RATE
        
        # Add normalized plant information to the list
        normalized_plant_info.extend([normalized_current_level, normalized_ramp_rate, min_level, max_level])
    
    # Combine all normalized components into a single array
    processed_state = np.array([normalized_time, normalized_demand, normalized_solar] + normalized_plant_info)
    
    processed_state = np.reshape(processed_state, (1, -1))
    return processed_state

def choose_action(state):
    # Assuming state contains information about current production levels for each plant
    # You need to preprocess the state to make it suitable for feeding into the policy network
    processed_state = preprocess_state(state)
    
    # Pass the processed state through the policy network to get action probabilities
    action_probabilities = policy_network(processed_state)
    
    # Initialize list to store actions for each power plant
    actions = []
    
    # Loop through each power plant
    for i, plant in enumerate(ud.plantInformation):
        current_level = plant['currentLevel']
        max_ramp_rate = plant['MaxRamp']
        MIN_PRODUCTION_LEVEL = plant['Min']
        MAX_PRODUCTION_LEVEL = plant['Max']
        
        # Calculate the allowable range of production levels based on the maximum ramp rate
        min_level = max(current_level - max_ramp_rate, MIN_PRODUCTION_LEVEL)
        max_level = min(current_level + max_ramp_rate, MAX_PRODUCTION_LEVEL)
        
        # Get the action probability distribution for this power plant
        plant_action_probabilities = action_probabilities[i]
        
        # Choose the action based on the probability distribution
        chosen_action = choose_action_from_distribution(plant_action_probabilities, min_level, max_level, max_ramp_rate, current_level)
        actions.append(chosen_action)
    
    return actions

def choose_action_from_distribution(action_probabilities, min_level, max_level, max_ramp_rate, current_level):
    # Discrete action selection based on action probabilities
    action = np.argmax(action_probabilities)  # Choose action with highest probability
    action = min(action, current_level+max_ramp_rate)
    action = max(action, current_level-max_ramp_rate)
    action = min(max(action, min_level), max_level)  # Ensure action is within allowable range
    return action

# Function to train the policy network
def train_step(states, joint_actions, rewards):
    with tf.GradientTape() as tape:
        logits = policy_network(states)
        loss = compute_loss(logits, joint_actions, rewards)
    gradients = tape.gradient(loss, policy_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables))

# Define loss function
def compute_loss(logits, actions, rewards):
    neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_log_prob * rewards)
    return loss

def get_state(current_time):
    # Get predicted demand and solar data for the current time
    predicted_demand = fgetElecDemand(current_time)
    solar_data = getSolarPred(current_time)
    
    # Get current levels and maximum ramp rates of all power plants
    plant_info = []
    for plant in sim.plantList:
        plant_info.append({
            "current_level": getSpecPlantInformation(plant)["CurrentLevel"],
            "max_ramp_rate": getSpecPlantInformation(plant)["MaxRamp"],
            "min_level": getSpecPlantInformation(plant)["Min"],
            "max_level": getSpecPlantInformation(plant)["Max"]
        })
    
    # Construct state array including time, demand, solar data, current levels, and maximum ramp rates
    state = [current_time, predicted_demand[0][0], solar_data[0][0]]
    for info in plant_info:
        state.extend([info["current_level"], info["max_ramp_rate"], info["min_level"], info["max_level"]])
    
    return np.array(state)

def calculate_reward(predDemand, currentTime):
    # Calculate the difference between current production and predicted demand
    production_diff = abs(getCurrentProduction(currentTime) - predDemand)
    
    # Efficiency reward based on how close the current production is to the maximum production
    if(production_diff == 0):
        efficiency_reward=.9
    else:
        efficiency_reward = 1/production_diff
    
    # Total reward as a combination of matching demand and efficiency
    total_reward = efficiency_reward
    
    return total_reward

def getDone(currentTime):
    if (currentTime < MAXTIME-1):
        return False 
    else:
        return True
    
# Function to interact with environment and collect experience
def run_episode(currentTime):
    state = get_state(currentTime)  # Reset environment and get initial state
    episode_reward = 0
    states, actions, rewards = [], [], []
    done = False
    while not done:
        action = choose_action(state)
        next_state = get_state(currentTime)
        reward = calculate_reward(next_state[1], currentTime)
        done =  getDone(currentTime)
        episode_reward += reward
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    return states, actions, rewards, episode_reward

# Training loop
num_episodes = 100
for episode in range(num_episodes):
    states, actions, rewards, episode_reward = run_episode(currentTime)

    # Calculate discounted rewards
    discounted_rewards = []
    cumulative_rewards = 0
    for reward in reversed(rewards):
        cumulative_rewards = reward + gamma * cumulative_rewards
        discounted_rewards.append(cumulative_rewards)
    discounted_rewards.reverse()

    # Normalize discounted rewards
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    # Convert lists to TensorFlow tensors
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

    # Train the policy network
    with tf.GradientTape() as tape:
        logits = policy_network(states)
        loss = compute_loss(logits, actions, discounted_rewards)
    gradients = tape.gradient(loss, policy_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables))

    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

xVals = []
production = []
predDemand = []
realDemand = []
plt.ion()
figure, ax = plt.subplots(figsize=(10, 8))
line1, = ax.plot(xVals, production, label="Total Production")
line2, = ax.plot(xVals, predDemand, label="Predicted Electricity Demand")
line3, = ax.plot(xVals, realDemand, label="Real Electricity Demand")


def livePlot(xVals, production, realDemand, currentTime):
    xVals.append(currentTime)
    production.append(los.logCurDict["totalProd"][currentTime])
    # print("CurPlotProd: " + str(los.logCurDict["totalProd"][currentTime]))
    predDemand.append(los.logFutDict["elecDem"][currentTime])
    realDemand.append(los.logCurDict["elecDem"][currentTime])

    '''
    xVals = xVals[-72: ]
    production = production[-72: ]
    predDemand = predDemand[-72: ]
    realDemand = realDemand[-72: ]
    '''

    # Set x-axis limit to show a specific range of x values
    if (currentTime <= 72):
        ax.set_xlim(min(xVals), max(xVals))
    else:
        ax.set_xlim(currentTime - 72, currentTime)

    # Set y-axis limit to show a specific range of y values
    ax.set_ylim(0, 4000)

    # print("Lengths - xVals:", len(xVals), "production:", len(production), "predDemand:", len(predDemand), "realDemand:", len(realDemand))

    # Update x-data
    line1.set_xdata(xVals)
    line2.set_xdata(xVals)
    line3.set_xdata(xVals)

    # Update y-data
    line1.set_ydata(production)
    line2.set_ydata(predDemand)
    line3.set_ydata(realDemand)

    plt.title("Production, Predicted Demand, Real Demand (MWh)")
    plt.legend(loc="upper left")
    figure.canvas.draw()
    figure.canvas.flush_events()
    # plt.pause(0.1)
    # time.sleep(.1)
    # https://www.youtube.com/watch?v=Ercd-Ip5PfQ


resetSimulation()
while currentTime != MAXTIME:
    futureAlgorithm(currentTime)

    currentAlgorithm(currentTime)
    livePlot(xVals, production, realDemand, currentTime)

    # print("Current Time: " + str(currentTime))
    currentTime = currentTime + 1

sim.saveAll()
