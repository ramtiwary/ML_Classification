# Databricks notebook source
print('Hello')

# COMMAND ----------


# Acept csv input
import pandas as pd

dbutils.widgets.text("input", "","")
dbutils.widgets.get("input")
FilePath = getArgument("input")
print ("Param -\'input':")
print (FilePath)

storage_account_name = "6threviewstorage"
storage_account_access_key = "kcDp67soI4Tpvlfeo3b8wesdlLBKnHTvJfdWTGFZPOn6+XgLfSWfL6mEKu05XCEzOgrwupoDzkzaSs63daxNIg=="

spark.conf.set(
 "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
 storage_account_access_key)


#file_location = "wasbs://example/location"+FilePath
file_location = "wasbs://reviewblob@6threviewstorage.blob.core.windows.net/output/houseprices.csv"
print(file_location)
file_type = "csv"


df = spark.read.format(file_type).option("inferSchema", "true").load(file_location)

#display(df.select("_c1"))

#df = spark.read.format('csv').load(“wasbs://ramblob@ramstorage12.blob.core.windows.net/Google_Stock_Price_Train.csv", inferSchema = True)
#df.format(“csv”)
df.show()




# COMMAND ----------

print(y)

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x_train = train.iloc[:,2].values
y_train = train.iloc[:,1].values

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

#y_pred = regressor.predict(x_test)

# COMMAND ----------


