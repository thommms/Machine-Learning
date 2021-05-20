from pyspark.sql import SparkSession

#after importing the spark session, lets create our spark program
spark = SparkSession.builder.appName("Task 2- Data correlation").getOrCreate()

#import the cleaned from task 1 dataset  and set header to be true to  properly format the column names
df = spark.read.csv('/user/tokonkw/Group6_Task_1_Output.csv',header =True, inferSchema=True)

df.show(10)

#calculate the correlation between Price and other fields

print("\ncorrelation between price and Distance = ",df.stat.corr('Price','Distance',method = 'pearson'))
print("\ncorrelation between price and Zipcode = ",df.stat.corr('Price','Zipcode',method = 'pearson'))
print("\ncorrelation between price and Bedroom = ",df.stat.corr('Price','#Bedroom',method = 'pearson'))
print("\ncorrelation between price and Bathroom = ",df.stat.corr('Price','#Bathroom',method = 'pearson'))
print("\ncorrelation between price and Car Garage = ",df.stat.corr('Price','#-Car Garage',method = 'pearson'))
print("\ncorrelation between price and Lot_size = ",df.stat.corr('Price','Lot_size',method = 'pearson'))
print("\ncorrelation between price and Property_count = ",df.stat.corr('Price','Property_count',method = 'pearson'))
print("\ncorrelation between price and Suburb = ",df.stat.corr('Price','Suburb_indexed',method = 'pearson'))
print("\ncorrelation between price and Type = ",df.stat.corr('Price','Type_indexed',method = 'pearson'))
print("\ncorrelation between price and Region_name = ",df.stat.corr('Price','Region_name_indexed',method = 'pearson'))

#we have calculated the correlation coefficient using pearson method in spark
#to calculate witht the remaining method, we have to use pandas

#first lets convert the dataframe to pandas
df_pandas= df.toPandas()

#lets remove the first column of the dataset since it's not useful for our coefficient correlation
pdf = df_pandas.iloc[:,1:]

#using price as the the goal or prediction value, we add it to a seperate dataframe
price_df = pdf["Price"]

print(price_df)

pdf.head(5)

print("===========================Pearson Method======================\n")
#calculate the correlation using the pearson method
df_pearson =pdf.corr(method="pearson")
print(df_pearson)

 #calculate the correllation coefficient with the pearson method
print("calculating price against other fields using pearson method\n")
print(pdf.corrwith(price_df,method="pearson",axis=0))

print("===========================Spearman Method======================\n")
#calculate the correlation using the spearman method
df_spearman = pdf.corr(method="spearman")
print(df_spearman)

#calculate the correllation coefficient with the spearman method
print("calculating price against other fields using Spearman method\n")
print(pdf.corrwith(price_df,method="spearman",axis=0))

print("===========================Kendall Method======================\n")
#calculate the correlation using the Kendall method
df_kendall = pdf.corr(method="kendall")
print("calculating price against other fields using kendall method\n")
print(df_kendall)

#calculate the correllation coefficient with the kendall method
print("calculating price against other fields using kendall method\n")
print(pdf.corrwith(price_df,method="kendall",axis=0))


#we have plotted 3 different method of correllation.
#If the goal is to predict what features will affect the price of the house or apartment, we can see from the three
#correllation method that number of bedroom "#Bedroom" will greatly affect the pricing.
