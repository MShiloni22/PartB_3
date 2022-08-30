from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import *
import os
import time
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import monotonically_increasing_id, desc


def learning_task(train_df, test_df):
    """
    Perform the learning task.
    Train a learning model on the train set, predict on the test set and report accuracy on test set
    :param train_df: Train set (spark.DataFrame)
    :param test_df: Test set (spark.DataFrame)
    :return: accuracy (float)
    """

    # Create the Logistic Regression, Random Forest models
    lr = LogisticRegression()
    rf = RandomForestClassifier()
    # Convert string column to categorical column
    model_indexer = StringIndexer(inputCol="Model", outputCol="model_index").setHandleInvalid("keep")
    device_indexer = StringIndexer(inputCol="Device", outputCol="device_index").setHandleInvalid("keep")
    user_indexer = StringIndexer(inputCol="User", outputCol="user_index").setHandleInvalid("keep")
    gt_indexer = StringIndexer(inputCol="gt", outputCol="label").setHandleInvalid("keep")
    # Create a one hot encoder
    model_encoder = OneHotEncoder(inputCol="model_index", outputCol="model_ohe")
    device_encoder = OneHotEncoder(inputCol="device_index", outputCol="device_ohe")
    user_encoder = OneHotEncoder(inputCol="user_index", outputCol="user_ohe")
    # Scale numeric features
    assembler1 = VectorAssembler(inputCols=["Creation_Time", "Arrival_Time", "x", "y", "z"], outputCol="features_unscaled")
    scaler = MinMaxScaler(inputCol="features_unscaled", outputCol="features_scaled")
    # Create a second assembler for the encoded columns
    assembler2 = VectorAssembler(inputCols=["model_ohe", "device_ohe", "user_ohe", "features_scaled"], outputCol="features")
    # Set up the pipeline
    pipeline_lr = Pipeline(stages=[assembler1, scaler, model_indexer, device_indexer, user_indexer, gt_indexer, model_encoder, device_encoder, user_encoder, assembler2, lr])
    pipeline_rf = Pipeline(stages=[assembler1, scaler, model_indexer, device_indexer, user_indexer, gt_indexer, model_encoder, device_encoder, user_encoder, assembler2, rf])

    # chose Random Forest since it has better performance than the model from section 2 (view PDF file for more details)
    predictions = pipeline_rf.fit(train_df).transform(test_df)
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    return accuracy


SCHEMA = StructType([StructField("Arrival_Time", LongType(), True),
                     StructField("Creation_Time", LongType(), True),
                     StructField("Device", StringType(), True),
                     StructField("Index", LongType(), True),
                     StructField("Model", StringType(), True),
                     StructField("User", StringType(), True),
                     StructField("gt", StringType(), True),
                     StructField("x", DoubleType(), True),
                     StructField("y", DoubleType(), True),
                     StructField("z", DoubleType(), True)])

spark = SparkSession.builder.appName('demo_app') \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()

os.environ['PYSPARK_SUBMIT_ARGS'] = \
    "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:2.4.8,com.microsoft.azure:spark-mssql-connector:1.0.1"
kafka_server = 'dds2020s-kafka.eastus.cloudapp.azure.com:9092'
topic = "activities"

streaming = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_server) \
    .option("subscribe", topic) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", False) \
    .option("maxOffsetsPerTrigger", 432) \
    .load() \
    .select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")).select("value.*")

query = streaming \
    .writeStream.queryName("input_df") \
    .format("memory") \
    .start()

# take initial data
time.sleep(5)
old_data = spark.sql("SELECT * FROM input_df")
old_data_rows_num = old_data.count()

for i in range(1, 31):

    # build train set from older data
    train_df = old_data

    # take the newest data arrived to build a test set
    time.sleep(5)
    old_and_new_data = spark.sql("SELECT * FROM input_df")
    old_and_new_data_rows_num = old_and_new_data.count()
    test_df = train_df.withColumn("MonIncID", monotonically_increasing_id())
    test_df = test_df.orderBy(desc("MonIncID")).drop("MonIncID").limit(old_and_new_data_rows_num - old_data_rows_num)

    # clean irrelevant columns and rows with gt=null
    train_df = train_df.select(["Creation_Time", "Arrival_Time", "Device", "Model", "User", "gt", "x", "y", "z"]).filter(train_df.gt != "null")
    test_df = test_df.select(["Creation_Time", "Arrival_Time", "Device", "Model", "User", "gt", "x", "y", "z"]).filter(test_df.gt != "null")

    # print iteration results
    print("--------------- iter = " + str(i) + " ---------------")
    print("Train set size: " + str(train_df.count()))
    print("Test set size: " + str(test_df.count()))
    acc = learning_task(train_df, test_df)
    print("Test accuracy :", acc)
    print()

    # set the train set for next iteration to be the train set + test set from current iteration
    old_data = old_and_new_data
    old_data_rows_num = old_and_new_data_rows_num
