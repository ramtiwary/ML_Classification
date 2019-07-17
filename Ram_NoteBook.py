# Databricks notebook source
# Creating widgets for leveraging parameters, and printing the parameters

#dbutils.widgets.text("input", "","")
#dbutils.widgets.get("input")
#y = getArgument("input")
#print ("Param -\'input':")
#print (y)

# Acept csv input
import pandas as pd

dbutils.widgets.text("input", "","")
dbutils.widgets.get("input")
FilePath = getArgument("input")
print ("Param -\'input':")
print (FilePath)

storage_account_name = "ramstorage12"
storage_account_access_key = "ZUPEADZeRTr0Z7th7OvhoX9w3h1yPtSm6ChW17JRMUAdE8rNzTD4AJplD3IZYejRjvCKP+zg0cRoC1S6kn1PaA=="

spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)


#file_location = "wasbs://example/location"+FilePath
file_location = "wasbs://ramblob@ramstorage12.blob.core.windows.net/Google_Stock_Price_Train.csv"
print(file_location)
file_type = "csv"


df = spark.read.format(file_type).option("inferSchema", "true").load(file_location)

#display(df.select("_c1"))

#df = spark.read.format('csv').load(“wasbs://ramblob@ramstorage12.blob.core.windows.net/Google_Stock_Price_Train.csv", inferSchema = True)
#df.format(“csv”)
df.show()

#dbutils.fs.mount(
#  source = "wasbs://ramblob@ramstorage12.blob.core.windows.net",
#  mount_point = "/mnt/<mount-name>",
#  extra_configs = {"<conf-key>":dbutils.secrets.get(scope = "<scope-name>", key = "<key-name>")})

#df = pd.read_csv(FilePath)
#df.head()




