
import org.apache.spark.sql._

val spark = SparkSession.builder.appName("CSV Reader").getOrCreate()

val iowa_df = spark.read.format("csv")
  .option("header", "true") // Use first line of all files as header
  .option("inferSchema", "true") // Automatically infer data types
  .load("/user/maria_dev/final/IowaCity/IowaCityWeather.csv")

val osh_df = spark.read.format("csv")
  .option("header", "true") // Use first line of all files as header
  .option("inferSchema", "true") // Automatically infer data types
  .load("/user/maria_dev/final/Oshkosh/OshkoshWeather.csv")

// 1
  // Create a date column
val osh_df_with_date = osh_df.withColumn("date", to_date(concat_ws("-", $"Year", $"Month", $"Day"), "yyyy-MM-dd"))

// Filter for days where the temperature is less than or equal to -10
// Coldest Day recorded in Wisconsin is -55F on 2/4/1996
// https://www.jsonline.com/story/weather/2023/10/20/wisconsin-weather-records-for-snow-cold-highest-temperature/71168205007/
val cold_days = osh_df_with_date.filter($"TemperatureF" <= -10 && $"TemperatureF" =!= null  && $"TemperatureF" > -56 ) 

// Filter for days where the temperature is greater than or equal to 95
val hot_days = osh_df_with_date.filter($"TemperatureF" >= 95 && $"TemperatureF" =!= null && $"TemperatureF" =!= "null" && $"TemperatureF" =!= "NA" && $"TemperatureF" > -56)

// Count the number of unique dates in each filtered DataFrame
val num_cold_days = cold_days.select("date").distinct().count()
val num_hot_days = hot_days.select("date").distinct().count()

if (num_cold_days > num_hot_days) {
    println("There are more cold days.")
    } else if (num_hot_days > num_cold_days) {
    println("There are more hot days.")
    } else {
    println("The number of cold and hot days is the same.")
}

//2

val monthToSeason = udf((month: Int) => month match {
  case 12 | 1 | 2 => "Winter"
  case 3 | 4 | 5 => "Spring"
  case 6 | 7 | 8 => "Summer"
  case _ => "Fall"
})

// Add a season column to each DataFrame
val osh_df_with_season = osh_df.withColumn("Season", monthToSeason($"Month"))
val iowa_df_with_season = iowa_df.withColumn("Season", monthToSeason($"Month"))

// Calculate the average season temperature for each DataFrame
val osh_avg_temp_by_season = osh_df_with_season.groupBy("Season").agg(avg("TemperatureF").alias("OshAvgTemp"))
val iowa_avg_temp_by_season = iowa_df_with_season.groupBy("Season").agg(avg("TemperatureF").alias("IowaAvgTemp"))

// Join the two DataFrames on the season column
  .withColumnRenamed("AvgTemp", "OshAvgTemp")


// Calculate the difference of temperatures and display it by season
val result_df = joined_df.withColumn("TempDifference", $"OshAvgTemp" - $"IowaAvgTemp")
result_df.show()


//

//3
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Window


val iowa_df= spark.sqlContext.read.option("header", "true").
    option("delimiter", ",").option("inferSchema", "true").option("mode", "PERMISSIVE")
    .csv("/user/maria_dev/final/IowaCity/IowaCityWeather.csv")

val osh_df = spark.sqlContext.read.option("header", "true").
    option("delimiter", ",").option("inferSchema", "true").option("mode", "PERMISSIVE")
    .csv("/user/maria_dev/final/Oshkosh/OshkoshWeather.csv")
    
val osh_df_with_date = osh_df.withColumn("date", to_date(concat_ws("-", $"Year", $"Month", $"Day"), "yyyy-MM-dd"))
val iowa_df_with_date =  iowa_df.withColumn("date", to_date(concat_ws("-", $"Year", $"Month", $"Day"), "yyyy-MM-dd"))
// Filter the datasets
val filtered_osh_df = osh_df_with_date.filter($"TemperatureF" >= -56 && $"TemperatureF" <= 115)
val filtered_iowa_df = iowa_df_with_date.filter($"TemperatureF" >= -56 && $"TemperatureF" <= 115)

// Concatenate the date and time columns into a single timestamp column
val osh_df_with_timestamp = filtered_osh_df.withColumn("Timestamp", expr("concat(date, ' ', TimeCST)"))
  .withColumn("Timestamp", unix_timestamp($"Timestamp", "yyyy-MM-dd hh:mm a").cast("timestamp"))
val iowa_df_with_timestamp = filtered_iowa_df.withColumn("Timestamp", expr("concat(date, ' ', TimeCST)"))
  .withColumn("Timestamp", unix_timestamp($"Timestamp", "yyyy-MM-dd hh:mm a").cast("timestamp"))

// Filter out the null values in the Timestamp column
val filtered_osh_df_with_timestamp = osh_df_with_timestamp.filter($"Timestamp".isNotNull)
val filtered_iowa_df_with_timestamp = iowa_df_with_timestamp.filter($"Timestamp".isNotNull)

// Convert the Timestamp column to a Unix timestamp
val osh_df_with_unix_timestamp = filtered_osh_df_with_timestamp.withColumn("UnixTime", unix_timestamp($"Timestamp"))
val iowa_df_with_unix_timestamp = filtered_iowa_df_with_timestamp.withColumn("UnixTime", unix_timestamp($"Timestamp"))


// Define the window
val window = Window.partitionBy(date_format($"Timestamp", "yyyy-MM-dd")).orderBy($"UnixTime").rangeBetween(0, 86400)

// Create a window based on the timestamp
val osh_df_with_max_min_temp_time =  osh_df_with_unix_timestamp.withColumn("MaxTemp", max("TemperatureF").over(window))
  .withColumn("MinTemp", min("TemperatureF").over(window))
  .withColumn("MaxTempTime", first(when($"TemperatureF" === $"MaxTemp", $"Timestamp")).over(window))
  .withColumn("MinTempTime", first(when($"TemperatureF" === $"MinTemp", $"Timestamp")).over(window))
val iowa_df_with_max_min_temp_time = iowa_df_with_unix_timestamp.withColumn("MaxTemp", max("TemperatureF").over(window))
  .withColumn("MinTemp", min("TemperatureF").over(window))
  .withColumn("MaxTempTime", first(when($"TemperatureF" === $"MaxTemp", $"Timestamp")).over(window))
  .withColumn("MinTempTime", first(when($"TemperatureF" === $"MinTemp", $"Timestamp")).over(window))

// Calculate the degrees difference
val osh_df_with_diff = osh_df_with_max_min_temp_time.withColumn("Diff", $"MaxTemp" - $"MinTemp")
val iowa_df_with_diff = iowa_df_with_max_min_temp_time.withColumn("Diff", $"MaxTemp" - $"MinTemp")

// Find the highest degrees difference and the time when the maximum and minimum temperatures occurred
val max_diff_osh = osh_df_with_diff.orderBy($"Diff".desc).select("Diff", "MaxTempTime", "MinTempTime").first()
val max_diff_iowa = iowa_df_with_diff.orderBy($"Diff".desc).select("Diff", "MaxTempTime", "MinTempTime").first()

// Compare the highest degrees difference in Oshkosh and Iowa City and print the city with the higher difference, the temperature difference, and when the maximum and minimum temperature occurred
if (max_diff_osh.getDouble(0) > max_diff_iowa.getDouble(0)) {
  println(s"The city with the highest amount of degrees difference is Oshkosh with a difference of ${max_diff_osh.getDouble(0)} degrees. The maximum temperature occurred at ${max_diff_osh.getTimestamp(1)} and the minimum temperature occurred at ${max_diff_osh.getTimestamp(2)}.")
} else if (max_diff_osh.getDouble(0) < max_diff_iowa.getDouble(0)) {
  println(s"The city with the highest amount of degrees difference is Iowa City with a difference of ${max_diff_iowa.getDouble(0)} degrees. The maximum temperature occurred at ${max_diff_iowa.getTimestamp(1)} and the minimum temperature occurred at ${max_diff_iowa.getTimestamp(2)}.")
} else {
  println("Both Oshkosh and Iowa City have the same highest amount of degrees difference.")
}

//4

import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Window

val iowa_df= spark.sqlContext.read.option("header", "true").
    option("delimiter", ",").option("inferSchema", "true").option("mode", "PERMISSIVE")
    .csv("/user/maria_dev/final/IowaCity/IowaCityWeather.csv")

val osh_df = spark.sqlContext.read.option("header", "true").
    option("delimiter", ",").option("inferSchema", "true").option("mode", "PERMISSIVE")
    .csv("/user/maria_dev/final/Oshkosh/OshkoshWeather.csv")
    
val osh_df_with_date = osh_df.withColumn("date", to_date(concat_ws("-", $"Year", $"Month", $"Day"), "yyyy-MM-dd"))
val iowa_df_with_date =  iowa_df.withColumn("date", to_date(concat_ws("-", $"Year", $"Month", $"Day"), "yyyy-MM-dd"))

// Filter the datasets, highest windspeed 135 7/4/1977
//https://www.fox6now.com/weather/worst-of-the-worst-of-wisconsins-severe-weather
val filtered_osh_df = osh_df_with_date.filter($"TemperatureF" >= -56 && $"TemperatureF" <= 115).filter($"`Wind SpeedMPH`" >= 0 && $"`Wind SpeedMPH`" <= 135)
val filtered_iowa_df = iowa_df_with_date.filter($"TemperatureF" >= -56 && $"TemperatureF" <= 115).filter($"Wind SpeedMPH" >= 0 && $"Wind SpeedMPH" <= 135)

// Concatenate the date and time columns into a single timestamp column
val osh_df_with_timestamp = filtered_osh_df.withColumn("Timestamp", expr("concat(date, ' ', TimeCST)"))
  .withColumn("Timestamp", unix_timestamp($"Timestamp", "yyyy-MM-dd hh:mm a").cast("timestamp"))
val iowa_df_with_timestamp = filtered_iowa_df.withColumn("Timestamp", expr("concat(date, ' ', TimeCST)"))
  .withColumn("Timestamp", unix_timestamp($"Timestamp", "yyyy-MM-dd hh:mm a").cast("timestamp"))

// Filter out the null values in the Timestamp column
val filtered_osh_df_with_timestamp = osh_df_with_timestamp.filter($"Timestamp".isNotNull)
val filtered_iowa_df_with_timestamp = iowa_df_with_timestamp.filter($"Timestamp".isNotNull)

// Convert the Timestamp column to a Unix timestamp
val osh_df_with_unix_timestamp = filtered_osh_df_with_timestamp.withColumn("UnixTime", unix_timestamp($"Timestamp"))
val iowa_df_with_unix_timestamp = filtered_iowa_df_with_timestamp.withColumn("UnixTime", unix_timestamp($"Timestamp"))

// Extract the month and hour from the timestamp
val osh_df_with_month_hour = osh_df_with_unix_timestamp.withColumn("Month", month($"Timestamp")).withColumn("Hour", hour($"Timestamp"))
val iowa_df_with_month_hour = iowa_df_with_unix_timestamp.withColumn("Month", month($"Timestamp")).withColumn("Hour", hour($"Timestamp"))

// Group by month and hour, and calculate the average temperature and wind speed
val osh_avg_df = osh_df_with_month_hour.groupBy("Month", "Hour").agg(avg("TemperatureF").alias("AvgTemp"), avg("`Wind SpeedMPH`").alias("AvgWindSpeed"))
val iowa_avg_df = iowa_df_with_month_hour.groupBy("Month", "Hour").agg(avg("TemperatureF").alias("AvgTemp"), avg("`Wind SpeedMPH`").alias("AvgWindSpeed"))

// Compute the absolute difference between the average temperature and 50 degrees
val osh_diff_df = osh_avg_df.withColumn("TempDiff", abs($"AvgTemp" - 50))
val iowa_diff_df = iowa_avg_df.withColumn("TempDiff", abs($"AvgTemp" - 50))

// Define a window partitioned by month and ordered by temperature difference and wind speed
val window = Window.partitionBy("Month").orderBy("TempDiff", "AvgWindSpeed")

// For each month, find the hour with the smallest temperature difference and lowest wind speed
val osh_best_hour_df = osh_diff_df.withColumn("Rank", rank().over(window)).filter($"Rank" === 1).drop("Rank")
val iowa_best_hour_df = iowa_diff_df.withColumn("Rank", rank().over(window)).filter($"Rank" === 1).drop("Rank")

// Add a "City" column to each DataFrame
val osh_best_hour_df_with_city = osh_best_hour_df.withColumn("City", lit("Oshkosh"))
val iowa_best_hour_df_with_city = iowa_best_hour_df.withColumn("City", lit("Iowa City"))

// Combine the data from both cities into one DataFrame
val combined_df = osh_best_hour_df_with_city.union(iowa_best_hour_df_with_city)

// Sort the DataFrame by month in ascending order
val sorted_df = combined_df.sort("Month")

// Rank the rows within each month based on the temperature difference and wind speed
val window = Window.partitionBy("Month").orderBy("TempDiff", "AvgWindSpeed")
val ranked_df = sorted_df.withColumn("Rank", rank().over(window))

// Filter for the rows with the lowest rank
val best_hour_df = ranked_df.filter($"Rank" === 1)

// Select the desired columns
val final_df = best_hour_df.select("City", "Month", "Hour", "AvgTemp", "TempDiff", "AvgWindSpeed")

// Display the final DataFrame
final_df.show()


import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Window

val iowa_df= spark.sqlContext.read.option("header", "true").
    option("delimiter", ",").option("inferSchema", "true").option("mode", "PERMISSIVE")
    .csv("/user/maria_dev/final/IowaCity/IowaCityWeather.csv")

val osh_df = spark.sqlContext.read.option("header", "true").
    option("delimiter", ",").option("inferSchema", "true").option("mode", "PERMISSIVE")
    .csv("/user/maria_dev/final/Oshkosh/OshkoshWeather.csv")
    
val osh_df_with_date = osh_df.withColumn("date", to_date(concat_ws("-", $"Year", $"Month", $"Day"), "yyyy-MM-dd"))
val iowa_df_with_date =  iowa_df.withColumn("date", to_date(concat_ws("-", $"Year", $"Month", $"Day"), "yyyy-MM-dd"))

// Filter the datasets, highest windspeed 135 7/4/1977
//https://www.fox6now.com/weather/worst-of-the-worst-of-wisconsins-severe-weather
val filtered_osh_df = osh_df_with_date.filter($"TemperatureF" >= -56 && $"TemperatureF" <= 115).filter($"`Wind SpeedMPH`" >= 0 && $"`Wind SpeedMPH`" <= 135)
val filtered_iowa_df = iowa_df_with_date.filter($"TemperatureF" >= -56 && $"TemperatureF" <= 115).filter($"Wind SpeedMPH" >= 0 && $"Wind SpeedMPH" <= 135)

// Concatenate the date and time columns into a single timestamp column
val osh_df_with_timestamp = filtered_osh_df.withColumn("Timestamp", expr("concat(date, ' ', TimeCST)"))
  .withColumn("Timestamp", unix_timestamp($"Timestamp", "yyyy-MM-dd hh:mm a").cast("timestamp"))
val iowa_df_with_timestamp = filtered_iowa_df.withColumn("Timestamp", expr("concat(date, ' ', TimeCST)"))
  .withColumn("Timestamp", unix_timestamp($"Timestamp", "yyyy-MM-dd hh:mm a").cast("timestamp"))

// Convert the timestamp to a numeric value
val osh_df_with_days = osh_df_with_timestamp.withColumn("Days", datediff(date_trunc("day", $"Timestamp"), lit("1970-01-01")))

// Create a 7-day window
val window = Window.orderBy(col("Days")).rangeBetween(0, 6)

// Calculate the average temperature in each window
val osh_df_with_avg_temp = osh_df_with_days.withColumn("AvgTemp", avg("TemperatureF").over(window))

// Find the maximum average temperature
val max_avg_temp = osh_df_with_avg_temp.select(max("AvgTemp")).first().getDouble(0)

// Find the start time of the 7-day period with the maximum average temperature
val start_time = osh_df_with_avg_temp.filter($"AvgTemp" === max_avg_temp).select(min("Timestamp")).first().getTimestamp(0)

println(s"The hottest average temperature over a 7-day period in Oshkosh is $max_avg_temp degrees. The 7-day period starts at $start_time.")

import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Window

val iowa_df= spark.sqlContext.read.option("header", "true").
    option("delimiter", ",").option("inferSchema", "true").option("mode", "PERMISSIVE")
    .csv("/user/maria_dev/final/IowaCity/IowaCityWeather.csv")

val osh_df = spark.sqlContext.read.option("header", "true").
    option("delimiter", ",").option("inferSchema", "true").option("mode", "PERMISSIVE")
    .csv("/user/maria_dev/final/Oshkosh/OshkoshWeather.csv")
    
val osh_df_with_date = osh_df.withColumn("date", to_date(concat_ws("-", $"Year", $"Month", $"Day"), "yyyy-MM-dd"))
val iowa_df_with_date =  iowa_df.withColumn("date", to_date(concat_ws("-", $"Year", $"Month", $"Day"), "yyyy-MM-dd"))

// Filter the datasets, highest windspeed 135 7/4/1977
//https://www.fox6now.com/weather/worst-of-the-worst-of-wisconsins-severe-weather
val filtered_osh_df = osh_df_with_date.filter($"TemperatureF" >= -56 && $"TemperatureF" <= 115).filter($"`Wind SpeedMPH`" >= 0 && $"`Wind SpeedMPH`" <= 135)
val filtered_iowa_df = iowa_df_with_date.filter($"TemperatureF" >= -56 && $"TemperatureF" <= 115).filter($"Wind SpeedMPH" >= 0 && $"Wind SpeedMPH" <= 135)

/// Convert 'TimeCST' to a timestamp format and extract the hour
val osh_df_with_hour = filtered_osh_df.withColumn("Hour", hour(to_timestamp($"TimeCST", "hh:mm a")))

// Group by year, month, day, and hour, and calculate the average temperature
val avg_temp_by_hour = osh_df_with_hour.groupBy($"Year", $"Month", $"Day", $"Hour").agg(avg("TemperatureF").as("AvgTemp"))

// Create a window partitioned by year, month, and day, and ordered by average temperature
val window = Window.partitionBy("Year", "Month", "Day").orderBy("AvgTemp")

// Rank the average temperatures within each day using dense_rank to allow for ties
val avg_temp_by_hour_with_rank = avg_temp_by_hour.withColumn("Rank", dense_rank().over(window))

// Filter the DataFrame to include only the rows with the lowest rank
val min_avg_temp_by_hour = avg_temp_by_hour_with_rank.filter($"Rank" === 1)

// Group by 'Hour' and count the number of occurrences of each hour
val hour_counts = min_avg_temp_by_hour.groupBy("Hour").count()

// Order by count in descending order and take the first row
val most_frequent_coldest_hour = hour_counts.orderBy(desc("count")).first()

println(s"The most frequent hour to be considered the coldest is ${most_frequent_coldest_hour.getAs[Int]("Hour")}, which occurs ${most_frequent_coldest_hour.getAs[Long]("count")} times.")