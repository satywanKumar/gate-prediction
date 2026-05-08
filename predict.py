import pickle
import pandas as pd

# ==============================
# 1. Load saved files
# ==============================
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# ==============================
# 2. New Student Data (INPUT)
# ==============================
data = {
    'age': 22,
    'gender': 'Male',
    'college_type': 'Private',
    'college_tier': 'Tier 2',
    'college_regular': 'Yes',
    'study_mode': 'Online',
    'preparation_type': 'Self Study',

    'daily_study_hours': 10,        # ✅ correct column
    'college_attendance_%': 75,    # ✅ missing column added

    'sem1': 70,
    'sem2': 72,
    'sem3': 74,
    'sem4': 75,
    'sem5': 76,
    'sem6': 78,
    'sem7': 80,
    'sem8': 82,

    'mock_test_1': 60,
    'mock_test_2': 65,
    'mock_test_3': 70,
    'mock_test_4': 75,
    'mock_test_5': 80
}

# ==============================
# 3. Convert to DataFrame
# ==============================
df = pd.DataFrame([data])

# ==============================
# 4. Apply Encoding
# ==============================
for col, le in encoders.items():
    df[col] = le.transform(df[col])

# ==============================
# 5. Feature Engineering (AUTO)
# ==============================
sem_cols = ['sem1','sem2','sem3','sem4','sem5','sem6','sem7','sem8']
mock_cols = ['mock_test_1','mock_test_2','mock_test_3','mock_test_4','mock_test_5']

df['avg_sem'] = df[sem_cols].mean(axis=1)
df['avg_mock'] = df[mock_cols].mean(axis=1)

# ==============================
# 6. Match Feature Order
# ==============================
df = df[features]

# ==============================
# 7. Prediction
# ==============================
prediction = model.predict(df)

print("🎯 Predicted GATE Score:", round(prediction[0], 2))