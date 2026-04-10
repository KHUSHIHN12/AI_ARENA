import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("final_bmtc_dataset.csv")  # change path if needed

# -------------------------------
# 1. Passenger Load (inside bus)
# -------------------------------
df["passenger_load"] = np.random.randint(10, 70, size=len(df))

# -------------------------------
# 2. Bus Capacity
# -------------------------------
df["bus_capacity"] = 60

# -------------------------------
# 3. Waiting Passengers (at stop)
# -------------------------------
df["waiting_passengers"] = np.random.randint(5, 50, size=len(df))

# -------------------------------
# 4. Boarding Time (depends on crowd)
# -------------------------------
df["boarding_time"] = df["waiting_passengers"] * 0.05  # simple model

# -------------------------------
# 5. Overcrowding Flag
# -------------------------------
df["overcrowding_flag"] = df["passenger_load"] > df["bus_capacity"]

# -------------------------------
# 6. Passenger Wait Time (IMPORTANT)
# -------------------------------
df["passenger_wait_time"] = df["headway"] * 0.5 + df["waiting_passengers"] * 0.1

# -------------------------------
# SAVE UPDATED DATASET
# -------------------------------
df.to_csv("updated_bmtc_dataset.csv", index=False)

print("✅ Dataset updated successfully")