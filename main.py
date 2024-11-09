import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv("./data.csv")

# Encode the 'Sex' column
label_encoder = LabelEncoder()
data["Sex"] = label_encoder.fit_transform(data["Sex"])

# Define features and target
X = data[["Age", "Fare", "Sex", "Pclass"]]
Y = data["2urvived"]  # Assuming the correct column name is 'Survived'

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, Y_train)

# Predict on test data
Y_predict = model.predict(X_test)

# Print confusion matrix
print(confusion_matrix(y_true=Y_test, y_pred=Y_predict))
