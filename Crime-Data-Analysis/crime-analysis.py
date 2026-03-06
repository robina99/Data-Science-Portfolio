# Re-run this cell
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
crimes = pd.read_csv("crimes.csv", dtype={"TIME OCC": str})
crimes.head(10)

crimes["TIME OCC"] = crimes["TIME OCC"].str.zfill(4)
crimes["HOUR"] = crimes["TIME OCC"].str[:2].astype(int)

peak_crime_hour = crimes["HOUR"].value_counts().idxmax()
print(peak_crime_hour)

# Which area has the largest frequency of night crimes
# Filter night crimes (10pm–3:59am)
night_crimes = crimes[
    (crimes["HOUR"] >= 22) | 
    (crimes["HOUR"] <= 3)
]

# Find area with highest frequency
peak_night_crime_location = night_crimes["AREA NAME"].value_counts().idxmax()
print(peak_night_crime_location)

#Identify the number of crimes committed against victims of different age groups.

# Define bins and labels
bins = [0, 17, 25, 34, 44, 54, 64, float("inf")]
labels = ["0-17", "18-25", "26-34", "35-44", 
          "45-54", "55-64", "65+"]

# Create age groups
age_groups = pd.cut(
    crimes["Vict Age"],
    bins=bins,
    labels=labels,
    right=True
)

# Count frequencies
victim_ages = age_groups.value_counts().sort_index()
print(victim_ages)

