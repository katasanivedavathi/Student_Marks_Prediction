
## how to know which algorithm is used in our project ##

import joblib
model=joblib.load('student_mark_predictor.pkl')
print(model)
print(type(model))