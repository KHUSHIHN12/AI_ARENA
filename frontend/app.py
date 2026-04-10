import streamlit as st
import pandas as pd
import numpy as np
import os

# -------------------------------
# LOAD DATA
# -------------------------------
file_path = os.path.join("..", "data", "expanded_bmtc_dataset.csv")
df = pd.read_csv(file_path)

st.set_page_config(layout="wide")
st.title("🚍 Dynamic Bus Scheduling Dashboard")

# -------------------------------
# SMART DECISION HANDLING
# -------------------------------
if "decision" not in df.columns:
    
    def temp_decision(row):
        if row["headway"] < 3:
            return "STOP_AT_NEXT_MAJOR_STOP"
        elif row["waiting_passengers"] > 30:
            return "CROWDED_STOP_INFO"
        else:
            return "STABLE"

    df["decision"] = df.apply(temp_decision, axis=1)
    df["hold_time"] = df["decision"].apply(lambda x: 120 if "STOP" in x else 0)

    st.info("⚙️ Using temporary decision logic")

else:
    if "hold_time" not in df.columns:
        df["hold_time"] = 120

    st.success("✅ Using decision engine output")

# -------------------------------
# KPI SECTION
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("🚍 Total Buses", df["bus_id"].nunique())
col2.metric("⚠️ Bunching Alerts", df[df["decision"].str.contains("STOP")].shape[0])
col3.metric("👥 Crowded Stops", df[df["decision"] == "CROWDED_STOP_INFO"].shape[0])
col4.metric("⏱ Avg Waiting Passengers", round(df["waiting_passengers"].mean(), 2))

st.markdown("---")

# -------------------------------
# 🚨 ALERT PANELS
# -------------------------------
st.markdown("## 🚨 System Alerts")

bunching_df = df[df["decision"].str.contains("STOP")]
crowd_df = df[df["decision"] == "CROWDED_STOP_INFO"]
stable_df = df[df["decision"] == "STABLE"]

# -------------------------------
# 🔴 BUNCHING PANEL
# -------------------------------
st.markdown("### 🔴 Bunching Alerts")

if bunching_df.empty:
    st.success("✅ No bunching detected")
else:
    for _, row in bunching_df.head(10).iterrows():
        st.error(f"""
🚨 **Bus {row['bus_id']} → {row['stop_name']}**

⚠️ Future Bunching Detected  
📊 Headway: {round(row['headway'], 2)} min  

👉 **ACTION: STOP at major stop**  
⏱ Hold Time: {row['hold_time']} sec  

**Reason:**  
- Headway Change: {round(row.get('headway_change_rate', 0), 2)}  
- Congestion: {row.get('congestion_level', 'N/A')}
""")

# -------------------------------
# 👥 CROWD PANEL
# -------------------------------
st.markdown("### 👥 Crowded Stops")

if crowd_df.empty:
    st.success("✅ No crowded stops")
else:
    for _, row in crowd_df.head(10).iterrows():
        st.warning(f"""
👥 **{row['stop_name']} Stop**

👥 Waiting Passengers: {row['waiting_passengers']}  

👉 STATUS: High Demand  
👉 ACTION: No schedule change  

Suggestion: Monitor or deploy extra bus
""")

# -------------------------------
# ✅ STABLE PANEL
# -------------------------------
st.markdown("### ✅ Stable Operations")

if stable_df.empty:
    st.warning("⚠️ No stable buses currently")
else:
    for _, row in stable_df.head(10).iterrows():
        st.success(f"""
✅ **Bus {row['bus_id']} → {row['stop_name']}**

System Stable  

📊 Headway: {round(row['headway'], 2)} min  
👥 Waiting Passengers: {row['waiting_passengers']}  

👉 ACTION: Maintain current schedule
""")

st.markdown("---")

# -------------------------------
# 📊 CHARTS SECTION
# -------------------------------
col1, col2 = st.columns(2)

# Passenger Demand Chart
with col1:
    st.subheader("📊 Passenger Demand by Stop")
    demand_df = df.groupby("stop_name")["waiting_passengers"].mean().reset_index()
    st.bar_chart(demand_df.set_index("stop_name"))

# Headway Graph
with col2:
    st.subheader("📈 Headway Trend")
    st.line_chart(df["headway"])