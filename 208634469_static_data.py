from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler, IndexToString
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, OneVsRest, LinearSVC, DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import warnings
import numpy as np
import os


SCHEMA = StructType([StructField("Arrival_Time",LongType(),True), 
                     StructField("Creation_Time",LongType(),True),
                     StructField("Device",StringType(),True), 
                     StructField("Index", LongType(), True),
                     StructField("Model", StringType(), True),
                     StructField("User", StringType(), True),
                     StructField("gt", StringType(), True),
                     StructField("x", DoubleType(), True),
                     StructField("y", DoubleType(), True),
                     StructField("z", DoubleType(), True)])

spark = SparkSession.builder.appName('demo_app')\
    .config("spark.kryoserializer.buffer.max", "512m")\
    .getOrCreate()

os.environ['PYSPARK_SUBMIT_ARGS'] = \
    "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:2.4.8,com.microsoft.azure:spark-mssql-connector:1.0.1"
kafka_server = 'dds2020s-kafka.eastus.cloudapp.azure.com:9092'
topic = "static"

static_df = spark.read\
                  .format("kafka")\
                  .option("kafka.bootstrap.servers", kafka_server)\
                  .option("subscribe", topic)\
                  .option("startingOffsets", "earliest")\
                  .option("failOnDataLoss",False)\
                  .option("maxOffsetsPerTrigger", 432)\
                  .load()\
                  .select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")).select("value.*")

warnings.filterwarnings("ignore")
df = static_df

# Create the Logistic Regression, Random Forest models
lr = LogisticRegression()
rf = RandomForestClassifier()  # better performance than lr
# Convert string column to categorical column
device_indexer = StringIndexer(inputCol="Device", outputCol="device_index").setHandleInvalid("keep")
user_indexer = StringIndexer(inputCol="User", outputCol="user_index").setHandleInvalid("keep")
gt_indexer = StringIndexer(inputCol="gt", outputCol="label").setHandleInvalid("keep")
# Create a one hot encoder
device_encoder = OneHotEncoder(inputCol="device_index", outputCol="device_ohe")
user_encoder = OneHotEncoder(inputCol="user_index", outputCol="user_ohe")
# Scale numeric features
assembler1 = VectorAssembler(inputCols=["Arrival_Time", "x", "y", "z"], outputCol="features_scaled1")
scaler = MinMaxScaler(inputCol="features_scaled1", outputCol="features_scaled")
# Create a second assembler for the encoded columns
assembler2 = VectorAssembler(inputCols=["device_ohe", "user_ohe", "features_scaled"], outputCol="features")
# Set up the pipeline
pipeline_lr = Pipeline(stages=[assembler1, scaler, device_indexer, user_indexer, gt_indexer, device_encoder, user_encoder, assembler2, lr])
pipeline_rf = Pipeline(stages=[assembler1, scaler, device_indexer, user_indexer, gt_indexer, device_encoder, user_encoder, assembler2, rf])


# Drop irrelevant columns and null labels
df = df.select(["Arrival_Time", "Device", "User", "gt", "x", "y", "z"]).filter(df.gt != "null")

# first split the data and train logistic regression on 50% of it and test 50% of it.
# then, take the data from the test set that was predicted to be in labels 4 or 5 (stairsup, stairsdown),
# train gradient boosted tree on 40% of it and test on the remaining 60%.
# in total we trained on 50% + 40% * 50% = 70% on the data, and predicted on 30%.
train, test = df.randomSplit([0.5, 0.5])
predictions = pipeline_lr.fit(train).transform(test)
predictions_nonstairs = predictions.select(["features", "prediction", "label", "gt"]) \
    .filter(predictions.prediction < 4)
predictions_stairs = predictions.select(["features", "prediction", "label", "gt"]) \
    .filter(predictions.prediction > 3) \
    .select(["features", "label", "gt"])
train, test = predictions_stairs.randomSplit([0.4, 0.6])
predictions_stairs = rf.fit(train).transform(test)
predictions_stairs = predictions_stairs.select(["features", "prediction", "label", "gt"])
predictions = predictions_nonstairs.union(predictions_stairs)


# Calculate accuracy
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Print results - visualization
print("label - labelIndex legend")
predictions.select(["gt", "label"]).distinct().orderBy(["label"]).show()

print("test accuracy:", round(accuracy, 2))
predictionAndLabels = predictions.rdd.map(lambda record: (record.prediction, record.label))
metrics = MulticlassMetrics(predictionAndLabels)
labels = predictions.rdd.map(lambda record: record.label).distinct().collect()
for label in sorted(labels):
    print("  * Class %s precision = %s" % (label, metrics.precision(label)))
    print("  * Class %s recall = %s" % (label, metrics.recall(label)))
confusion_matrix = metrics.confusionMatrix().toArray()
print("Confusion matrix (predicted classes are in columns, ordered by class label asc, true classes are in rows):")
print(np.array(confusion_matrix).astype(int))