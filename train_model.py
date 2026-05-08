import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_excel("gate_realistic_dataset.xlsx")

# ==============================
# 2. Handle Categorical Columns
# ==============================
categorical_cols = [
    'gender',
    'college_type',
    'college_tier',
    'college_regular',
    'study_mode',
    'preparation_type'
]

le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # save encoder

# ==============================
# 3. Feature Engineering
# ==============================

# Average semester marks
sem_cols = ['sem1','sem2','sem3','sem4','sem5','sem6','sem7','sem8']
df['avg_sem'] = df[sem_cols].mean(axis=1)

# Average mock test marks
mock_cols = ['mock_test_1','mock_test_2','mock_test_3','mock_test_4','mock_test_5']
df['avg_mock'] = df[mock_cols].mean(axis=1)

# ==============================
# 4. Features & Target
# ==============================
X = df.drop(['student_id', 'final_gate_score'], axis=1)
y = df['final_gate_score']

# ==============================
# 5. Train Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. Model Training
# ==============================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==============================
# 7. Save Everything
# ==============================

# Model
pickle.dump(model, open("model.pkl", "wb"))

# Feature names (VERY IMPORTANT)
pickle.dump(X.columns.tolist(), open("features.pkl", "wb"))

# Encoders (for future UI/API use)
pickle.dump(le_dict, open("encoders.pkl", "wb"))

print("✅ Model trained successfully!")
print("📦 Files saved:")
print(" - model.pkl")
print(" - features.pkl")
print(" - encoders.pkl")