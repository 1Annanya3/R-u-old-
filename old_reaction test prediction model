import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


#Connect to Google Sheets using service account
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client = gspread.authorize(creds)

spreadsheet = client.open("CS3237_Health_Assessment_Data")
sheet = spreadsheet.worksheet("Reaction Time Test") #"Reaction Time Test" worksheet

data = sheet.get_all_records()
df = pd.DataFrame(data)

required_columns = ["participant_gender", "reaction_time_ms", "actual_age"]
assert all(col in df.columns for col in required_columns), "Missing required columns."


#Data cleaning
df["reaction_time_ms"] = df["reaction_time_ms"].astype(str).str.replace(",", "", regex=False).astype(float)
df["actual_age"] = df["actual_age"].astype(float).astype(int)

#Regression

X = df[["participant_gender", "reaction_time_ms"]]
y = df["actual_age"]

#encode gender column (categorical)
preprocessor = ColumnTransformer(
    transformers=[
        ("gender_encoder", OneHotEncoder(categories=[["M", "F"]], drop='first'), ["participant_gender"]),
        ("passthrough", "passthrough", ["reaction_time_ms"])
    ]
)

#Regression Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),  # Polynomial features
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

print("Model training complete.")
print("Model test score (R²):", model.score(X_test, y_test))

#Prediction function
def predict_age(gender, reaction_time):
    # gender must be "M" or "F"
    input_data = pd.DataFrame([[gender, reaction_time]], columns=["participant_gender", "reaction_time_ms"])
    predicted_age = model.predict(input_data)[0]
    return predicted_age

#Example prediction
example_gender = "F"
example_reaction_time = 403.4  # ms

predicted = predict_age(example_gender, example_reaction_time)
print(f"Predicted Age for {example_gender} with {example_reaction_time} ms reaction time = {predicted:.2f} years")
