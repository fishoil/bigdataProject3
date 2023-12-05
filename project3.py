from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
import pandas as pd
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def convert_to_libsvm_format(df, label_column):
    """
    Converts a pandas dataframe to LibSVM format with floating-point values.
    """
    libsvm_data = []
    for _, row in df.iterrows():
        # Ensure label is a float (1.0 for Malignant, 0.0 for Benign)
        label = 1.0 if row[label_column] == 'M' else 0.0
        libsvm_row = [str(label)]
        for i, value in enumerate(row.drop(label_column)):
            # Convert feature values to float
            libsvm_row.append(f"{i + 1}:{float(value)}")
        libsvm_data.append(" ".join(libsvm_row))
    return libsvm_data


# Initialize SparkSession for CSV reading
spark = SparkSession.builder.appName("CancerDiagnosis").getOrCreate()

# Read the CSV file using Pandas
df = pd.read_csv("[replace]")

# Convert the DataFrame to LibSVM format
libsvm_data = convert_to_libsvm_format(df.drop(['id'], axis=1), 'diagnosis')  # Drop 'id' column as it's not a feature

# Save the LibSVM data to a file
libsvm_file_path = "[replace]"
with open(libsvm_file_path, "w") as file:
    for line in libsvm_data:
        file.write(line + "\n")

# Stop SparkSession
spark.stop()

# Initialize SparkContext for MLlib
sc = SparkContext('local[*]', appName="CancerDiagnosis")

# Load and parse the data in LibSVM format
data = MLUtils.loadLibSVMFile(sc, libsvm_file_path)

# Split the data into training and test sets (70% training, 30% testing)
trainingData, testData = data.randomSplit([0.7, 0.3])

# Train a RandomForest model
rfModel = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                       numTrees=30, featureSubsetStrategy="auto",
                                       impurity='gini', maxDepth=4, maxBins=32)

# Train a Logistic Regression model
lrModel = LogisticRegressionWithLBFGS.train(trainingData, numClasses=2)

# Common evaluation function for both models
def evaluate_rf_model(model, testData):
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    metrics = MulticlassMetrics(labelsAndPredictions)
    print("Random Forest Model Evaluation")
    print("Accuracy: {:.2f}%".format(100 * metrics.accuracy))
    print("F1 Score:", metrics.fMeasure(1.0))
    print("Precision:", metrics.precision(1.0))
    print("Recall:", metrics.recall(1.0))
    metrics = BinaryClassificationMetrics(labelsAndPredictions)
    print("Area under ROC:", metrics.areaUnderROC)


def evaluate_lr_model(model, testData):
    predictionAndLabels = testData.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    metrics = MulticlassMetrics(predictionAndLabels)
    print("Logistic Regression Model Evaluation")
    print("Accuracy: {:.2f}%".format(100 * metrics.accuracy))
    print("F1 Score:", metrics.fMeasure(1.0))
    print("Precision:", metrics.precision(1.0))
    print("Recall:", metrics.recall(1.0))
    # ROC Curve
    metrics = BinaryClassificationMetrics(predictionAndLabels)
    print("Area under ROC:", metrics.areaUnderROC)

# Evaluate Random Forest Model
evaluate_rf_model(rfModel, testData)

# Evaluate Logistic Regression Model
print()
print()
evaluate_lr_model(lrModel, testData)

# Stop SparkContext
sc.stop()