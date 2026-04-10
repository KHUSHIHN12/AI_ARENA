import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="BunchGuard Dashboard", page_icon="🚍", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .big-font { font-size:20px !important; font-weight: bold; }
    .stAlert { padding: 10px; }
</style>
""", unsafe_allow_html=True)

# --- SIMULATED DATASET GENERATION ---
def get_simulation_data():
    np.random.seed(int(time.time()))
    data = {
        'bus_id': ['B1', 'B2', 'B3', 'B4', 'B5'],
        'stop_name': ['Central Stn', 'Oak St', 'Main Ave', 'High Park', 'Terminal'],
        'arrival_time': ['10:00', '10:05', '10:12', '10:20', '10:30'],
        'headway': [6, 2, 8, 3, 5], # Minutes between buses
        'headway_change_rate': [0.1, -0.5, 0.2, -0.8, 0.0], # Negative = closing gap
        'passengers': [45, 95, 30, 110, 50],
        'demand_level': ['Medium', 'High', 'Low', 'High', 'Medium'],
        'bunching_label': [0, 1, 0, 2, 0], # 0=Safe, 1=Risk, 2=Bunching
        'traffic_delay': [2, 5, 1, 8, 2]
    }
    df = pd.DataFrame(data)
    # Map bunching labels to colors
    df['status_color'] = df['bunching_label'].map({0: '🟢 Safe', 1: '🟡 Risk', 2: '🔴 Bunching'})
    return df

# --- SESSION STATE FOR SIMULATION ---
if 'df' not in st.session_state:
    st.session_state.df = get_simulation_data()

# --- HEADER ---
st.title("🚍 BunchGuard: Dynamic Bus Scheduling & Anti-Bunching System")
st.markdown("---")

# --- SIDEBAR - CONTROLS ---
st.sidebar.header("🎛️ Simulation Controls")
if st.sidebar.button("Simulate Next Time Step"):
    st.session_state.df = get_simulation_data()
    st.rerun()

traffic_level = st.sidebar.slider("Traffic Congestion Level", 0, 100, 50)
bus_speed = st.sidebar.select_slider("Bus Frequency Setting", options=["Low", "Normal", "High"], value="Normal")

# --- TABS ---
tab1, tab2 = st.tabs(["👨‍💼 Admin Panel", "👥 Passenger Panel"])

# ==========================================
# 👨‍💼 ADMIN PANEL
# ==========================================
with tab1:
    st.header("Control System Monitor")
    
    # 1. LIVE MONITORING TABLE
    st.subheader("Live Bus Operations")
    
    def color_status(val):
        color = 'white'
        if 'Bunching' in val: color = '#ffcccc' # Light Red
        elif 'Risk' in val: color = '#fff3cd' # Light Yellow
        elif 'Safe' in val: color = '#d4edda' # Light Green
        return f'background-color: {color}'

    styled_df = st.session_state.df[['bus_id', 'stop_name', 'headway', 'passengers', 'demand_level', 'status_color']]
    st.dataframe(styled_df.style.applymap(color_status, subset=['status_color']), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    
    # 2. EARLY WARNING PANEL
    with col1:
        st.subheader("⚠️ Early Warning System")
        risky_buses = st.session_state.df[st.session_state.df['headway_change_rate'] < 0]
        if not risky_buses.empty:
            for _, row in risky_buses.iterrows():
                st.warning(f"Bus {row['bus_id']} closing gap at {row['stop_name']}. Headway: {row['headway']} min")
        else:
            st.success("No bunching risk detected.")

    # 3. AI DECISION & 4. INSERTION PANEL
    with col2:
        st.subheader("🤖 AI Action Panel")
        for _, row in st.session_state.df.iterrows():
            if row['headway'] <= 3 and row['demand_level'] == 'Low':
                st.info(f"👉 Hold {row['bus_id']} at {row['stop_name']} for 30s (Low Demand)")
            elif row['headway'] <= 3 and row['demand_level'] == 'High':
                st.success(f"✅ {row['bus_id']} {row['stop_name']}: Maintain speed (High Demand)")
            
            if row['passengers'] > 100:
                st.error(f"🚨 ALERT: Insert extra bus for {row['bus_id']} at {row['stop_name']}!")

    # 5. OVERTAKING & 6. DYNAMIC SCHEDULING
    with col3:
        st.subheader("📊 Dynamic Scheduling")
        
        # Overtaking Logic
        for i in range(len(st.session_state.df)-1):
            if st.session_state.df.iloc[i]['passengers'] > 90 and st.session_state.df.iloc[i+1]['passengers'] < 50:
                st.warning(f"🔄 Allow Overtaking: {st.session_state.df.iloc[i+1]['bus_id']} to pass {st.session_state.df.iloc[i]['bus_id']}")

        # Scheduling Logic
        if bus_speed == "High":
            st.metric("Frequency", "10 buses/hour", "+40%")
        elif bus_speed == "Low":
            st.metric("Frequency", "4 buses/hour", "-20%")
        else:
            st.metric("Frequency", "7 buses/hour", "0%")

    # 7. ANALYTICS
    st.markdown("---")
    st.subheader("📊 Performance Analytics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Waiting Time (System On)", "3.2 min", "-1.5 min")
    c2.metric("Bunching Incidents", "1", "-2 vs yesterday")
    c3.metric("Passenger Comfort Index", "85%", "+10%")
    
    # Simple chart
    st.line_chart(st.session_state.df.set_index('bus_id')['headway'])

# ==========================================
# 👥 PASSENGER PANEL
# ==========================================
with tab2:
    st.header("Live Passenger Information")
    
    # 1. BUS ARRIVAL INFO
    st.subheader("Next Bus Arrivals")
    
    for _, row in st.session_state.df.head(3).iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])
            col1.markdown(f"**Bus {row['bus_id']}** - {row['stop_name']}")
            
            # Load Status
            if row['passengers'] > 90:
                col2.markdown("🔴 **Crowded**")
                advice = "Suggest: Wait for next bus"
            elif row['passengers'] > 50:
                col2.markdown("🟡 **Moderate**")
                advice = "You can board"
            else:
                col2.markdown("🟢 **Comfortable**")
                advice = "You can board"
            
            col3.markdown(f"Arrival: {np.random.randint(1,10)} min")
            st.caption(f"Status: {advice}")
            st.markdown("---")

    # 4. LIVE TRACKING (SIMULATED)
    st.subheader("Live Map Simulation")
    # Simulate bus positions
    map_data = pd.DataFrame({
        'lat': [37.77, 37.78, 37.79, 37.80],
        'lon': [-122.41, -122.42, -122.43, -122.44],
        'bus': ['B1', 'B2', 'B3', 'B4']
    })
    st.map(map_data)

st.sidebar.markdown("---")
st.sidebar.caption("BunchGuard AI - Hackathon Demo 2026")
