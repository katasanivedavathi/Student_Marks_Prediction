import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# sample training data
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [35,40,50,55,60,65,70,75,80,85]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df[["Marks"]]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "student_mark_predictor.pkl")

print("Model trained & saved successfully")
