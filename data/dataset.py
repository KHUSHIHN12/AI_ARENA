import numpy as np
import pandas as pd

def generate_bmtc_data(num_buses=6, num_trips=4):
    np.random.seed(42)

    # Real Bengaluru stops
    stops = ["Shivajinagar","MG Road","Trinity","Ulsoor",
             "KR Puram","Marathahalli","Bellandur",
             "Silk Board","HSR Layout","Electronic City"]

    # 🆕 Stop IDs
    stop_ids = {stop: f"S{i+1}" for i, stop in enumerate(stops)}

    base_time = [4,3,3,5,6,5,4,6,5,4]
    distance = [2,1.5,1.2,2.5,3,2.2,2,3.5,2.5,3]

    # Traffic signals
    signals = [1,2,1,2,3,2,2,4,2,1]

    # Bus frequency
    dispatch_gap = 8  # minutes
    bus_frequency_per_hour = int(60 / dispatch_gap)

    data = []

    # Traffic model
    def traffic_model(time):
        hour = (6 + int(time // 60)) % 24
        if 8 <= hour <= 11 or 17 <= hour <= 20:
            return 3.0, "High"
        elif 12 <= hour <= 16:
            return 1.8, "Medium"
        else:
            return 1.0, "Low"

    for trip in range(num_trips):
        for bus in range(1, num_buses + 1):

            current_time = trip * 120 + bus * dispatch_gap
            prev_delay = 0

            for i, stop in enumerate(stops):

                # Traffic
                factor, level = traffic_model(current_time)

                # Delay
                traffic_delay = np.random.lognormal(0.6, 0.5) * factor
                signal_delay = signals[i] * np.random.uniform(0.5, 1.5)

                delay = traffic_delay + signal_delay
                travel_time = base_time[i] * factor + delay

                # Passenger behavior
                passengers = np.random.randint(10, 80)
                dwell_time = (passengers * 0.12) + np.random.uniform(0.5, 1.5)

                arrival_time = current_time + travel_time
                departure_time = arrival_time + dwell_time

                data.append({
                    "bus_id": f"B{bus}",

                    # ✅ stop id + name
                    "stop_id": stop_ids[stop],
                    "stop_name": stop,

                    "arrival_time": round(arrival_time, 2),
                    "departure_time": round(departure_time, 2),

                    # delays
                    "traffic_delay": round(traffic_delay, 2),
                    "signal_delay": round(signal_delay, 2),
                    "total_delay": round(delay, 2),
                    "previous_delay": round(prev_delay, 2),

                    # passengers
                    "passengers": passengers,
                    "dwell_time": round(dwell_time, 2),

                    # traffic
                    "congestion_level": level,
                    "traffic_factor": factor,

                    # route
                    "distance_to_next_stop": distance[i],
                    "num_signals": signals[i],

                    # frequency
                    "bus_frequency_per_hour": bus_frequency_per_hour
                })

                current_time = departure_time
                prev_delay = delay

    df = pd.DataFrame(data)

    # -------------------------------
    # HEADWAY
    # -------------------------------
    df = df.sort_values(by=["stop_id", "arrival_time"])
    df["headway"] = df.groupby("stop_id")["arrival_time"].diff()
    df["headway"] = df["headway"].fillna(10)

    # -------------------------------
    # HEADWAY CHANGE RATE
    # -------------------------------
    df["headway_change_rate"] = df.groupby("stop_id")["headway"].diff().fillna(0)

    # -------------------------------
    # DEMAND LEVEL
    # -------------------------------
    def demand_level(p):
        if p > 60:
            return "High"
        elif p > 30:
            return "Medium"
        else:
            return "Low"

    df["demand_level"] = df["passengers"].apply(demand_level)

    # -------------------------------
    # LABEL
    # -------------------------------
    def label(h):
        if h > 7:
            return 0   # Safe
        elif h > 3:
            return 1   # Risk
        else:
            return 2   # Bunching

    df["bunching_label"] = df["headway"].apply(label)

    # -------------------------------
    # SAVE
    # -------------------------------
    df.to_csv("final_bmtc_dataset.csv", index=False)

    print("✅ Final dataset created successfully!")
    print("📊 Bus Frequency:", bus_frequency_per_hour, "buses/hour")
    print(df.head())

    return df


# Run
dataset = generate_bmtc_data()