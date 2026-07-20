import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("data/students.csv")

X = data[
    ["Attendance",
     "Internal",
     "Assignment",
     "Maths",
     "CN",
     "ML",
     "Communication"]
]

y = data["Result"]

model = DecisionTreeClassifier()
model.fit(X, y)

student = [[85,80,78,70,75,82,76]]

prediction = model.predict(student)

print("Prediction:", prediction[0])