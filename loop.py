import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd

# Step 1: Define the Fuzzy Control System

# New Antecedent/Consequent objects hold universe variables and membership functions
blood_pressure = ctrl.Antecedent(np.arange(80, 181, 1), 'blood_pressure')
oxygen_saturation = ctrl.Antecedent(np.arange(90, 101, 1), 'oxygen_saturation')
body_temperature = ctrl.Antecedent(np.arange(35, 40, 0.1), 'body_temperature')
heart_healthiness = ctrl.Consequent(np.arange(0, 101, 1), 'heart_healthiness')

# Auto-membership function population is possible with .automf(3, 5, or 7)
blood_pressure.automf(3)
oxygen_saturation.automf(3)
body_temperature.automf(3)

# Custom membership functions can be built interactively with a familiar Pythonic API
heart_healthiness['low'] = fuzz.trimf(heart_healthiness.universe, [0, 25, 50])
heart_healthiness['medium'] = fuzz.trimf(heart_healthiness.universe, [30, 50, 70])
heart_healthiness['high'] = fuzz.trimf(heart_healthiness.universe, [50, 75, 100])

# Rules
rule1 = ctrl.Rule(blood_pressure['poor'] | oxygen_saturation['poor'] | body_temperature['poor'], heart_healthiness['low'])
rule2 = ctrl.Rule(oxygen_saturation['average'], heart_healthiness['medium'])
rule3 = ctrl.Rule(blood_pressure['good'] & oxygen_saturation['good'] & body_temperature['good'], heart_healthiness['high'])

heart_health_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

# Step 2: Read Input Data from Excel
df = pd.read_excel('heart_health_data.xlsx')

# Prepare the output column
heart_healthiness_levels = []

# Step 3: Compute Heart Healthiness for Each Record
for index, row in df.iterrows():
    heart_health_sim = ctrl.ControlSystemSimulation(heart_health_ctrl)
    heart_health_sim.input['blood_pressure'] = row['BloodPressure']
    heart_health_sim.input['oxygen_saturation'] = row['OxygenSaturation']
    heart_health_sim.input['body_temperature'] = row['BodyTemperature']
    heart_health_sim.compute()
    heart_healthiness_levels.append(heart_health_sim.output['heart_healthiness'])

# Add the computed results to the DataFrame
df['HeartHealthiness'] = heart_healthiness_levels

# Step 4: Write Results to an Excel File
df.to_excel('heart_health_results.xlsx', index=False)

print('Heart healthiness levels have been computed and saved.')
