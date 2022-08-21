from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import *
import os
import time


def learning_task(df):
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    # Create the logistic regression model
    lr = LogisticRegression()

    # Convert string column to categorical column
    # indexer = StringIndexer(
    #   inputCols=["Device", "User", "gt"],
    #   # need to add explanation why we deleted column 'Model': doesn't have two distinct values
    #   outputCols=["device_index", "user_index", "label"])
    device_indexer = StringIndexer(inputCol="Device", outputCol="device_index")
    user_indexer = StringIndexer(inputCol="User", outputCol="user_index")
    gt_indexer = StringIndexer(inputCol="gt", outputCol="label")

    # We create a one hot encoder
    # encoder = OneHotEncoder(inputCols=["device_index", "user_index"],
    #                       outputCols=["device_ohe", "user_ohe"])
    device_encoder = OneHotEncoder(inputCol="device_index", outputCol="device_ohe")
    user_encoder = OneHotEncoder(inputCol="user_index", outputCol="user_ohe")

    # Input list for scaling
    inputs = ["Arrival_Time", "Creation_Time", "x", "y", "z"]

    # We scale our inputs
    assembler1 = VectorAssembler(inputCols=inputs, outputCol="features_scaled1")
    scaler = MinMaxScaler(inputCol="features_scaled1", outputCol="features_scaled")

    # We create a second assembler for the encoded columns
    assembler2 = VectorAssembler(inputCols=["device_ohe", "user_ohe", "features_scaled"],
                                 outputCol="features")

    # Create stages list
    myStages = [assembler1, scaler,
                device_indexer, user_indexer, gt_indexer,
                device_encoder, user_encoder,
                assembler2, lr]

    # Set up the pipeline
    pipeline = Pipeline(stages=myStages)

    # split to two folds, will be used as train/test alternately
    fold1, fold2 = df.randomSplit([0.5, 0.5], seed=12345)

    # combine fit, transform and evaluation in a loop for both learning procedures
    accuracies = []
    for train_fold, test_fold in zip([fold1, fold2], [fold2, fold1]):
        # We fit the model using the training data
        pModel = pipeline.fit(train_fold)

        # We transform the data
        testingPred = pModel.transform(test_fold)

        # Evaluate with accuracy
        testingPred = testingPred.select("features", "label", "prediction")
        testingPred = testingPred.withColumn("LabEqPred",
                                                 (testingPred.label == testingPred.prediction).cast('double'))
        test_accuracy = (testingPred.groupBy().sum().collect()[0][2]) / testingPred.count()
        accuracies.append(test_accuracy)

    return sum(accuracies) / len(accuracies)


SCHEMA = StructType([StructField("Arrival_Time", LongType(),True),
                     StructField("Creation_Time", LongType(),True),
                     StructField("Device", StringType(),True),
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
topic = "activities"

streaming = spark.readStream\
                  .format("kafka")\
                  .option("kafka.bootstrap.servers", kafka_server)\
                  .option("subscribe", topic)\
                  .option("startingOffsets", "earliest")\
                  .option("failOnDataLoss", False)\
                  .option("maxOffsetsPerTrigger", 432)\
                  .load()\
                  .select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")).select("value.*")

learningTask = streaming\
  .writeStream.queryName("input_df")\
  .format("memory")\
  .outputMode("append")\
  .start()

time.sleep(30)
for x in range(5):
    time.sleep(1)
    df = spark.sql("SELECT * FROM input_df")
    print("iter=", x)
    print("Average accuracy over 2-folds of whole data:", learning_task(df))
