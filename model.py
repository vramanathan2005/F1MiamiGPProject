import fastf1
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# Enable FastF1 cache
fastf1.Cache.enable_cache("/Users/varunramanathan/Downloads/f1data_cache")

# Load laps
all_laps = []
for year in range(2021, 2026):
    races = ['Jeddah', 'Baku']
    if year >= 2022:
        races += ['Miami', 'Singapore']
    if year >= 2023:
        races.append('Las Vegas')

    for race in races:
        try:
            session = fastf1.get_session(year, race, 'R')
            session.load()
            laps = session.laps.copy()
            laps['Circuit'] = race
            all_laps.append(laps)
        except Exception as e:
            print(f"Skipped {race} {year}: {e}")

laps = pd.concat(all_laps, ignore_index=True)

# Feature extraction
def extract_overtake_data(laps):
    examples = []
    drivers = laps['Driver'].unique()

    for driver in drivers:
        driver_laps = laps[laps['Driver'] == driver].sort_values('LapNumber').copy()
        driver_laps['NextPos'] = driver_laps['Position'].shift(-1)
        driver_laps['GapToAhead'] = driver_laps['LapStartTime'].diff().dt.total_seconds()
        driver_laps['LapTimeSeconds'] = driver_laps['LapTime'].dt.total_seconds()
        driver_laps['RollingAvgLapTime'] = driver_laps['LapTimeSeconds'].rolling(3).mean()

        for _, row in driver_laps.iterrows():
            if pd.isna(row['Position']) or pd.isna(row['NextPos']):
                continue
            if pd.isna(row['LapTime']) or pd.isna(row['TyreLife']) or pd.isna(row['Compound']) or pd.isna(row['Stint']):
                continue

            lap_time = row['LapTimeSeconds']
            improved = 1 if int(row['NextPos']) < int(row['Position']) else 0

            examples.append({
                'LapTime': lap_time,
                'RollingAvgLapTime': row['RollingAvgLapTime'] if pd.notna(row['RollingAvgLapTime']) else lap_time,
                'TyreLife': row['TyreLife'],
                'Position': int(row['Position']),
                'Stint': row['Stint'],
                'Compound': row['Compound'],
                'GapToAhead': row['GapToAhead'] if pd.notna(row['GapToAhead']) else 0,
                'ImprovedNextLap': improved,
                'Circuit': row['Circuit']
            })

    return pd.DataFrame(examples)

df = extract_overtake_data(laps)
print("Extracted examples:", len(df))
print(df.head())

# Encode categorical variables
df['Compound'] = df['Compound'].fillna('UNKNOWN')
le = LabelEncoder()
df['CompoundEncoded'] = le.fit_transform(df['Compound'])

df = pd.get_dummies(df, columns=['Circuit'], prefix='Circuit')

# Feature matrix and labels
features = ['LapTime', 'RollingAvgLapTime', 'TyreLife', 'Position', 'Stint', 'GapToAhead', 'CompoundEncoded'] + \
           [col for col in df.columns if col.startswith('Circuit_')]

X = df[features]
y = df['ImprovedNextLap']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ----------- Random Forest -----------
rf_model = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\nRandomForest Evaluation:")
print(classification_report(y_test, rf_preds, digits=4))
print("Accuracy:", accuracy_score(y_test, rf_preds))

# Save model
joblib.dump(rf_model, "model_rf_overtake.pkl")

# ----------- XGBoost -----------
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08,
                              subsample=0.9, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss',
                              random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

print("\nXGBoost Evaluation:")
print(classification_report(y_test, xgb_preds, digits=4))
print("Accuracy:", accuracy_score(y_test, xgb_preds))

# Save model
joblib.dump(xgb_model, "model_xgb_overtake.pkl")

print("Both models saved successfully.")
