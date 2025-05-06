import fastf1
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
cache_dir = os.path.join(os.getcwd(), "f1data_cache")
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)


# ---------- CONFIG ----------
st.set_page_config(page_title="Ferrari Strategy Analyzer – Miami GP 2025", layout="wide")

# ---------- STYLING ----------
custom_css = """
<style>
body {
    background-color: #f5f7fa;
    color: #1e1e1e;
    font-family: 'Inter', sans-serif;
}
.driver-ferrari { color: #DC0000; font-weight: 600; }
.driver-mercedes { color: #00D2BE; font-weight: 600; }
.driver-williams { color: #005AFF; font-weight: 600; }
.container-box {
    background-color: #ffffff;
    border: 1px solid #e3e6ea;
    border-radius: 8px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------- LOAD DATA ----------
try:
    session = fastf1.get_session(2025, 'Miami', 'R')
    session.load()
except Exception as e:
    st.error(f"Could not load 2025 Miami GP session: {e}")
    st.stop()

model_path = "model_rf_overtake.pkl"
if not os.path.exists(model_path):
    st.error("Pre-trained model not found. Please train and save it as model_rf_overtake.pkl")
    st.stop()

model = joblib.load(model_path)
laps = session.laps

# ---------- PLOTTING UTILITY ----------
def plot_gap_by_distance(chaser, target, lap):
    c_data = chaser[lap - 1:lap].get_car_data().add_distance()
    t_data = target[lap - 1:lap].get_car_data().add_distance()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_data['Distance'], t_data['Speed'], label='Target', color='black', alpha=0.5)
    ax.plot(c_data['Distance'], c_data['Speed'], label='Chaser', color='red', alpha=0.6)

    min_len = min(len(t_data), len(c_data))
    ax.fill_between(
        t_data['Distance'][:min_len],
        (t_data['Speed'][:min_len] - c_data['Speed'][:min_len]),
        alpha=0.3, color='purple', label='Delta Zone'
    )

    sector_distances = [1100, 2800]
    for i, dist in enumerate(sector_distances):
        ax.axvline(dist, linestyle='--', color='gray', alpha=0.3)
        ax.text(dist, ax.get_ylim()[1]*0.95, f"S{i+1}", rotation=90, va='top', ha='center', fontsize=9)

    drs_zones = [(80, 240), (1300, 1470)]
    for dz in drs_zones:
        ax.axvspan(*dz, color='aqua', alpha=0.2, label='DRS Zone' if 'DRS Zone' not in ax.get_legend_handles_labels()[1] else '')

    overtaking_zones = [(250, 400), (1300, 1500)]
    for oz in overtaking_zones:
        ax.axvspan(*oz, color='limegreen', alpha=0.2, label='Overtaking Zone' if 'Overtaking Zone' not in ax.get_legend_handles_labels()[1] else '')

    ax.set_title(f"Speed Delta on Track – Lap {lap}")
    ax.set_xlabel("Distance on Lap (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.legend()
    st.pyplot(fig)

# ---------- MAIN APP ----------
st.header("Ferrari Strategy Simulation – Miami GP 2025")

# === Scenario Simulation ===
st.subheader("Strategy What-If Simulator")

tabs = st.tabs([
    "Swap Hamilton ahead of Leclerc to chase Antonelli",
    "Swap Leclerc back ahead of Hamilton to chase Antonelli",
    "Avoid early swap: would Sainz have passed Leclerc?"
])

tab_scenarios = [
    {"chaser": "HAM", "target": "ANT", "lap_range": (30, 40), "tab": tabs[0]},
    {"chaser": "LEC", "target": "ANT", "lap_range": (45, 55), "tab": tabs[1]},
    {"chaser": "SAI", "target": "LEC", "lap_range": (30, 38), "tab": tabs[2]},
]




def predict_and_plot(chaser_code, target_code, lap, circuit_name):
    ch = laps[laps['Driver'] == chaser_code]
    tg = laps[laps['Driver'] == target_code]
    ch_lap = ch[ch['LapNumber'] == lap]
    tg_lap = tg[tg['LapNumber'] == lap]

    if ch_lap.empty or tg_lap.empty:
        st.warning(f"No lap data available for {chaser_code} or {target_code} at lap {lap}")
        return

    # Construct base feature row
    feat = {
        'LapTime': ch_lap['LapTime'].values[0] / pd.Timedelta(seconds=1) if pd.notna(ch_lap['LapTime'].values[0]) else 0,
        'TyreLife': ch_lap['TyreLife'].values[0],
        'Position': ch_lap['Position'].values[0],
        'Stint': ch_lap['Stint'].values[0],
        'CompoundEncoded': 0,  # will set below
        'GapToAhead': (tg_lap['LapStartTime'].values[0] - ch_lap['LapStartTime'].values[0]) / pd.Timedelta(seconds=1)
    }

    # Compound encoding
    from sklearn.preprocessing import LabelEncoder
    all_compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET', 'UNKNOWN']
    le = LabelEncoder()
    le.fit(all_compounds)
    comp = ch_lap['Compound'].values[0] if pd.notna(ch_lap['Compound'].values[0]) else 'UNKNOWN'
    feat['CompoundEncoded'] = le.transform([comp])[0]

    # Load expected feature columns from model training
    model = joblib.load("model_rf_overtake.pkl")
    expected_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []

    # Create a full input row
    input_df = pd.DataFrame([feat])

    # Add one-hot encoding for circuit
    for col in expected_cols:
        if col.startswith("Circuit_"):
            input_df[col] = 1 if col == f'Circuit_{circuit_name}' else 0

    # Add any missing columns (in case of legacy model)
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]  # ensure column order

    # Make prediction
    prob = model.predict_proba(input_df)[0][1]

    # Display
    st.metric(f"{chaser_code} overtaking {target_code} (Lap {lap})", f"{prob:.0%}")
    st.caption(f"Visualizing {chaser_code} vs {target_code} – Lap {lap}")
    plot_gap_by_distance(session.laps.pick_driver(chaser_code), session.laps.pick_driver(target_code), lap)

    # Interpretation logic
    if prob >= 0.7:
        interpretation = (
            f"Based on the model, it is very likely that {chaser_code} could have overtaken {target_code} on lap {lap}. "
            "The chaser likely had a clear pace and tyre advantage in key zones."
        )
    elif prob >= 0.4:
        interpretation = (
            f"There was a moderate chance that {chaser_code} could have passed {target_code} on lap {lap}. "
            "Some favorable conditions existed, but it was not a high-confidence situation."
        )
    elif prob >= 0.15:
        interpretation = (
            f"It is unlikely that {chaser_code} had a realistic opportunity to overtake {target_code} on lap {lap}. "
            "The pace differential and overtaking setup were probably insufficient."
        )
    else:
        interpretation = (
            f"The model suggests a very low probability of {chaser_code} overtaking {target_code} on lap {lap}. "
            "Maintaining position was likely the best available strategy."
        )

    st.markdown(f"**Interpretation:** {interpretation}")
def show_probability_table(chaser, target, laps, model, circuit_name, lap_range):
    from sklearn.preprocessing import LabelEncoder

    with st.expander(f"Model probabilities for {chaser} overtaking {target} (Laps {lap_range[0]}–{lap_range[1]})"):
        prob_table = []

        for lap in range(lap_range[0], lap_range[1] + 1):
            ch_lap = laps[(laps['Driver'] == chaser) & (laps['LapNumber'] == lap)]
            tg_lap = laps[(laps['Driver'] == target) & (laps['LapNumber'] == lap)]

            if ch_lap.empty or tg_lap.empty:
                continue

            feat = {
                'LapTime': ch_lap['LapTime'].values[0] / pd.Timedelta(seconds=1) if pd.notna(ch_lap['LapTime'].values[0]) else 0,
                'TyreLife': ch_lap['TyreLife'].values[0],
                'Position': ch_lap['Position'].values[0],
                'Stint': ch_lap['Stint'].values[0],
                'CompoundEncoded': 0,
                'GapToAhead': (tg_lap['LapStartTime'].values[0] - ch_lap['LapStartTime'].values[0]) / pd.Timedelta(seconds=1)
            }

            all_compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET', 'UNKNOWN']
            le = LabelEncoder()
            le.fit(all_compounds)
            comp = ch_lap['Compound'].values[0] if pd.notna(ch_lap['Compound'].values[0]) else 'UNKNOWN'
            feat['CompoundEncoded'] = le.transform([comp])[0]

            input_df = pd.DataFrame([feat])
            for col in model.feature_names_in_:
                if col.startswith("Circuit_"):
                    input_df[col] = 1 if col == f'Circuit_{circuit_name}' else 0
            for col in model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model.feature_names_in_]

            prob = model.predict_proba(input_df)[0][1]
            prob_table.append({"Lap": lap, "Probability": f"{prob:.0%}"})

        st.dataframe(pd.DataFrame(prob_table))

for sc in tab_scenarios:
    with sc["tab"]:
        swap_lap = st.slider(
            f"When would Ferrari swap drivers? ({sc['chaser']} ahead of {sc['target']})",
            *sc["lap_range"],
            key=f"{sc['chaser']}_{sc['target']}"
        )

        st.write(f"Simulating overtakes for {sc['chaser']} vs {sc['target']} in laps after lap {swap_lap}...")

        for lap in range(swap_lap + 1, swap_lap + 6):  # Show next 5 laps after swap
            st.markdown(f"---\n### Lap {lap}")
            predict_and_plot(sc["chaser"], sc["target"], lap, "Miami")


