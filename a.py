
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder #increasing dimentionality ,convert categorical variables into a binary (0 or 1) representation
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow.sklearn
import joblib


 
df=pd.read_csv("airbnb_listings.csv")
x=df.drop(columns=["ListingID","PricePerNight"])
y=df["PricePerNight"]

Preprocessor=ColumnTransformer(
    [("cat",OneHotEncoder(handle_unknown='ignore'),["City","RoomType"])],
    remainder="passthrough"
)

model=Pipeline([
    ("preprocessor",Preprocessor),
    ("regression",RandomForestRegressor(n_estimators=100,random_state=42))]
)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
  
model.fit(x_train,y_train)
joblib.dump(model,"airbnb.pkl")
print("model saved succesfully")


