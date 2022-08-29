from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import *
import os
import time
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import numpy as np


def learning_task(df):
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

    # Split to two folds, will be used as train/test alternately
    fold1, fold2 = df.randomSplit([0.5, 0.5], seed=12345)

    # Combine fit, transform and evaluation in a loop for both learning procedures
    accuracies = []
    """
    for train_fold, test_fold in zip([fold1, fold2], [fold2, fold1]):
        # Fit the model using the train dataset
        pModel = pipeline.fit(train_fold)

        # Transform the test dataset
        testingPred = pModel.transform(test_fold)

        # Evaluate with accuracy
        testingPred = testingPred.select("features", "label", "prediction")
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(testingPred)
        accuracies.append(accuracy)
    """
    for train, test in zip([fold1, fold2], [fold2, fold1]):
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

        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        accuracies.append(accuracy)

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

    return sum(accuracies) / len(accuracies)


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

for i in range(1, 11):
    time.sleep(30)
    df = spark.sql("SELECT * FROM input_df")
    df = df.select(["Arrival_Time", "Device", "User", "gt", "x", "y", "z"]).filter(df.gt != "null")
    print("iter = " + str(i) + ", aggregated number of records is " + str(df.count()), end=", ")
    acc = learning_task(df)
    print("average accuracy in 2-folds-cross-validation over the whole data:", acc)
