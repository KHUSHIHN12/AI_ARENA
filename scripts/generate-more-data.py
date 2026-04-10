import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# -------------------------------
# PATH SETUP
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "updated_bmtc_dataset.csv")

# Load existing dataset
print("📂 Loading existing dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded: {len(df)} rows")

# Get statistics from existing data
print("\n📊 Analyzing data distributions...")

# Define bus IDs and stop IDs
bus_ids = [f"B{i}" for i in range(1, 11)]  # B1 to B10
stop_ids = [f"S{i}" for i in range(1, 6)]   # S1 to S5
stop_names = ["Shivajinagar", "Marathahalli", "Whitefield", "Electronics City", "Koramangala"]

# Extract numeric features statistics
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
stats = {}
for col in numeric_cols:
    stats[col] = {
        'mean': df[col].mean(),
        'std': df[col].std(),
        'min': df[col].min(),
        'max': df[col].max()
    }

print(f"✅ Extracted statistics for {len(numeric_cols)} numeric columns")

# Generate synthetic data
print("\n🔄 Generating synthetic data (1000 more samples)...")
num_samples = 1000
synthetic_data = []

for _ in range(num_samples):
    row = {
        'bus_id': np.random.choice(bus_ids),
        'stop_id': np.random.choice(stop_ids),
        'stop_name': np.random.choice(stop_names),
    }
    
    # Generate time data
    arrival = np.random.uniform(0, 60)
    departure = arrival + np.random.uniform(2, 15)
    row['arrival_time'] = arrival
    row['departure_time'] = departure
    
    # Generate delay features with realistic constraints
    traffic_delay = np.random.normal(stats['traffic_delay']['mean'], stats['traffic_delay']['std'])
    traffic_delay = np.clip(traffic_delay, stats['traffic_delay']['min'], stats['traffic_delay']['max'])
    row['traffic_delay'] = max(0, traffic_delay)
    
    signal_delay = np.random.normal(stats['signal_delay']['mean'], stats['signal_delay']['std'])
    signal_delay = np.clip(signal_delay, stats['signal_delay']['min'], stats['signal_delay']['max'])
    row['signal_delay'] = max(0, signal_delay)
    
    row['total_delay'] = row['traffic_delay'] + row['signal_delay'] + np.random.normal(0, 0.5)
    row['total_delay'] = max(0, row['total_delay'])
    
    row['previous_delay'] = np.random.uniform(stats['previous_delay']['min'], stats['previous_delay']['max'])
    
    # Generate passenger data
    row['passengers'] = int(np.random.normal(stats['passengers']['mean'], stats['passengers']['std']))
    row['passengers'] = np.clip(row['passengers'], 10, 100)
    
    row['dwell_time'] = np.random.normal(stats['dwell_time']['mean'], stats['dwell_time']['std'])
    row['dwell_time'] = max(0.5, row['dwell_time'])
    
    # Congestion level
    row['congestion_level'] = np.random.choice(['Low', 'Medium', 'High'])
    
    row['traffic_factor'] = np.random.uniform(0.8, 2.0)
    row['distance_to_next_stop'] = np.random.uniform(1, 5)
    row['num_signals'] = int(np.random.uniform(0, 4))
    row['bus_frequency_per_hour'] = np.random.uniform(4, 10)
    row['headway'] = np.random.uniform(4, 15)
    row['headway_change_rate'] = np.random.uniform(-5, 5)
    
    # Demand level
    row['demand_level'] = np.random.choice(['Low', 'Medium', 'High'])
    row['bunching_label'] = int(np.random.choice([0, 1], p=[0.7, 0.3]))
    
    row['passenger_load'] = int(np.random.normal(stats['passenger_load']['mean'], stats['passenger_load']['std']))
    row['passenger_load'] = np.clip(row['passenger_load'], 10, 70)
    
    row['bus_capacity'] = 60
    
    row['waiting_passengers'] = int(np.random.normal(stats['waiting_passengers']['mean'], stats['waiting_passengers']['std']))
    row['waiting_passengers'] = np.clip(row['waiting_passengers'], 0, 50)
    
    row['boarding_time'] = np.random.uniform(0.5, 5)
    row['overcrowding_flag'] = bool(np.random.choice([False, True], p=[0.6, 0.4]))
    
    row['passenger_wait_time'] = np.random.uniform(1, 15)
    
    synthetic_data.append(row)

synthetic_df = pd.DataFrame(synthetic_data)
print(f"✅ Generated {len(synthetic_df)} synthetic samples")

# Combine original and synthetic data
print("\n📦 Combining original and synthetic data...")
combined_df = pd.concat([df, synthetic_df], ignore_index=True)
print(f"✅ Combined dataset size: {len(combined_df)} rows (was {len(df)})")

# Save expanded dataset
expanded_path = os.path.join(BASE_DIR, "data", "expanded_bmtc_dataset.csv")
combined_df.to_csv(expanded_path, index=False)
print(f"\n✅ Expanded dataset saved to: {expanded_path}")

# Show statistics
print(f"\n📊 Dataset Comparison:")
print(f"   Original: {len(df)} rows")
print(f"   Synthetic: {len(synthetic_df)} rows")
print(f"   Combined: {len(combined_df)} rows (4.17x increase)")
print(f"\n✨ Data generation complete! Ready for training with expanded dataset.")
