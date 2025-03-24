from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, avg, when, dense_rank
from pyspark.sql.window import Window

# Initialize Spark Session with GCS support
spark = SparkSession.builder \
    .appName("GCS_StudentDetails_Transformation") \
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
    .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", "C:/Users/MT24054/Downloads/modak-nabu-7103e448ac3f (1) (1).json") \
    .getOrCreate()

# Set legacy parser policy to avoid timestamp format issues
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# GCS path containing all CSV files
gcs_path = "gs://your-bucket-name/student_data/*.csv"  # All CSVs in one read

# Subject columns
subject_cols = ["Math", "English", "Hindi", "Science", "Economics", "History", "Geography", "Telugu", "Physics", "Chemistry"]

# PostgreSQL DB connection details
db_url = "jdbc:postgresql://w3.training5.modak.com:5432/postgres"
db_props = {
    "user": "mt24054",
    "password": "mt24054@m10y24",
    "driver": "org.postgresql.Driver"
}

# Read and combine all CSV files
df = spark.read.option("header", True).option("inferSchema", True).csv(gcs_path)
df = df.fillna(0, subset=subject_cols)

# Student Performance Transformation
df_perf = df.withColumn("total_marks", sum([col(c) for c in subject_cols])) \
            .withColumn("subjects_count", sum([when(col(c) > 0, 1).otherwise(0) for c in subject_cols])) \
            .withColumn("percentage", (col("total_marks") / (col("subjects_count") * 100)) * 100) \
            .withColumn("grade", when(col("percentage") >= 90, "A+")
                                  .when(col("percentage") >= 80, "A")
                                  .when(col("percentage") >= 70, "B")
                                  .when(col("percentage") >= 60, "C")
                                  .otherwise("D"))

window_section = Window.partitionBy("Class", "Section").orderBy(col("total_marks").desc())
df_perf = df_perf.withColumn("section_rank", dense_rank().over(window_section))

# Class Averages
df_class_avg = df.groupBy("Class").agg(
    *[avg(col(c)).alias(f"avg_{c}") for c in subject_cols]
)

df_class_avg = df_class_avg.withColumn(
    "overall_class_percentage",
    (sum([col(f"avg_{c}") for c in subject_cols]) / (len(subject_cols) * 100)) * 100
)

# School-wide Averages
df_school_avg = df.select(subject_cols).agg(
    *[avg(col(c)).alias(f"school_avg_{c}") for c in subject_cols]
)

# Class Toppers
window_class = Window.partitionBy("Class").orderBy(col("percentage").desc())
df_topper = df_perf.withColumn("class_rank", dense_rank().over(window_class)) \
                   .filter(col("class_rank") == 1) \
                   .select("Name", "ID", "Class", "Section", "percentage")

# Write Results to PostgreSQL
df_perf.write.jdbc(db_url, "transformed_combined_performance", mode="overwrite", properties=db_props)
df_class_avg.write.jdbc(db_url, "transformed_combined_class_avg", mode="overwrite", properties=db_props)
df_school_avg.write.jdbc(db_url, "transformed_combined_school_monitoring", mode="overwrite", properties=db_props)
df_topper.write.jdbc(db_url, "transformed_combined_class_toppers", mode="overwrite", properties=db_props)

print("✅ All files combined, processed, and saved to PostgreSQL successfully.")
