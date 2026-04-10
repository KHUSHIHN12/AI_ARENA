from pathlib import Path

import pandas as pd


def decision_engine(row):
    """
    Processes real-time telemetry to prevent bunching at major transit hubs.
    """
    major_stops = ["Shivajinagar", "KR Puram", "Marathahalli", "Silk Board", "Electronic City"]
    bunching_threshold = 3.0
    crowd_threshold = 30

    bus_id = row["bus_id"]
    stop = row["stop_name"]
    headway = row["headway"]
    waiting_pax = row["waiting_passengers"]
    h_rate = row["headway_change_rate"]
    congestion = row["congestion_level"]
    p_load = row["passenger_load"]

    is_bunching_risk = headway < bunching_threshold
    is_high_demand = waiting_pax > crowd_threshold
    is_major_stop = stop in major_stops

    decision = "STABLE"
    hold_time = 0
    alert = ""

    if is_bunching_risk:
        decision = "CONTROL ACTION"
        hold_time = 120

        context = f"Headway {headway}m < {bunching_threshold}m."
        if h_rate < 0:
            context += f" Gap shrinking (Rate: {h_rate})."
        if congestion == "High":
            context += " High traffic context."

        location_action = f"HOLD AT {stop}" if is_major_stop else "HOLD AT NEXT MAJOR STOP"
        alert = (
            f"ALERT: {bus_id} at {stop}: {location_action} ({hold_time}s). "
            f"Waiting: {waiting_pax}. {context}"
        )
    elif is_high_demand:
        decision = "INFORMATION ALERT"
        alert = (
            f"INFO: {stop} is crowded ({waiting_pax} pax). "
            f"Bus {bus_id} load: {p_load}. No bunching risk."
        )
    else:
        alert = f"OK: {bus_id} at {stop}: Operations normal."

    return pd.Series(
        [decision, hold_time, alert],
        index=["decision", "hold_time", "alert_message"],
    )


def run_decision_engine():
    """
    Loads the updated BMTC dataset, applies the decision engine, and returns
    the enriched dataframe for downstream use.
    """
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "updated_bmtc_dataset.csv"
    df = pd.read_csv(dataset_path)

    decision_results = df.apply(decision_engine, axis=1)
    final_df = pd.concat([df, decision_results], axis=1)

    critical_columns = ["bus_id", "stop_name", "headway", "decision", "alert_message"]
    critical_rows = final_df[final_df["decision"] != "STABLE"][critical_columns]
    print(critical_rows)

    return final_df


if __name__ == "__main__":
    run_decision_engine()
