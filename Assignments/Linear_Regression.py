import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Monthly_Income": [20000, 25000, 30000, 35000, 40000, 50000, 60000, 70000, 75000, 80000]
}

df = pd.DataFrame(data)

X = df[["Experience"]]
y = df["Monthly_Income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X)

plt.scatter(df["Experience"], df["Monthly_Income"], color='blue', label="Actual Data")
plt.plot(df["Experience"], y_pred, color='red', label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Monthly Income")
plt.title("Simple Linear Regression: Experience vs Monthly Income")
plt.legend()
plt.show()
