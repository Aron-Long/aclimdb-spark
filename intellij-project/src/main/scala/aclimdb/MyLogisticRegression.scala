package aclimdb

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object MyLogisticRegression {

  def getTime(startTime: Long): Double = (System.nanoTime - startTime) / 1e9

  def logisticRegression(resizedTrainingData: DataFrame, resizedTestData: DataFrame, spark: SparkSession): Unit = {
    //Logistic regression
    import spark.implicits._

    val startTime = System.nanoTime
    var lr = new LogisticRegression()
      .setMaxIter(40)
      .setRegParam(0.01)
    val model = lr.fit(resizedTrainingData)
    val trainedTestData = model.transform(resizedTestData)
    val mse = trainedTestData
      .map(row => row.getAs[Double]("label") - row.getAs[Double]("prediction"))
      .map (x => x * x)
      .reduce(_+_) / trainedTestData.count

    println("Mean squared error by logistic regression: " + mse)
    println("Logistic regression ran in " + getTime(startTime) + " seconds")
  }

  def decisionTree(resizedTrainingData: DataFrame, resizedTestData: DataFrame, spark: SparkSession): Unit = {
    import spark.implicits._

    //Decision tree
    val startTime = System.nanoTime

    val model = DecisionTree.trainRegressor(
      resizedTrainingData
        .rdd
        .map(row =>
          new org.apache.spark.mllib.regression.LabeledPoint(
            row.getAs[Double]("label"),
            org.apache.spark.mllib.linalg.SparseVector.fromML(row.getAs[SparseVector]("features"))
          )
        ),
      Map[Int, Int](), "variance",
      5,
      32)
    val mse = resizedTestData.map(row =>
      row.getAs[Double]("label") -
        model.predict(row.getAs[org.apache.spark.mllib.linalg.SparseVector]("features"))

    ).map(x => x * x).reduce(_ + _) / resizedTestData.count
    println("Mean squared error by decision tree regressor: " + mse)
    println("Decision tree regressor ran in " + getTime(startTime) + " seconds")
  }

  def elasticNetLogisticRegression(resizedTrainingData: DataFrame, resizedTestData: DataFrame, spark: SparkSession): Unit = {
    //ElasticNet regression
    import spark.implicits._

    val startTime = System.nanoTime
    var lr = new LogisticRegression()
      .setMaxIter(40)
      .setRegParam(0.01)
      .setElasticNetParam(0.1)
    val model = lr.fit(resizedTrainingData)
    val trainedTestData = model.transform(resizedTestData)
    val mse = trainedTestData
      .map(row => row.getAs[Double]("label") - row.getAs[Double]("prediction"))
      .map (x => x * x)
      .reduce(_+_) / trainedTestData.count

    println("Mean squared error by elastic net logistic regression: " + mse)
    println("Elastic net logistic regression ran in " + getTime(startTime) + " seconds")
  }

  def randomForest(resizedTrainingData: DataFrame, resizedTestData: DataFrame, spark: SparkSession) : Unit = {
    import spark.implicits._
    val startTime = System.nanoTime
    val rf = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("features").setNumTrees(30)
    val model = rf.fit(resizedTrainingData)
    val mse = model
      .transform(resizedTestData)
      .map(row => row.getAs[Double]("label") - row.getAs[Double]("prediction"))
      .map(x => x * x)
      .reduce(_ + _) / resizedTestData.count

    println("Mean squared error by random forest regressor: " + mse)
    println("Random forest regressor ran in " + getTime(startTime) + " seconds")
  }

  def main(args: Array[String]): Unit = {
    //turn off log
//    Logger.getRootLogger.setLevel(Level.WARN)

    println("Start program. Start spark session...")
    val conf = new SparkConf()
//      .setMaster("local[2]")
      .setAppName("aclimdb")
    val sc = new SparkContext(conf).setLogLevel("WARN")
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("MyLogisticRegression")
      .getOrCreate()
    import spark.implicits._
    println("Spark session inited. Load data ...")

    //Load data
    val Seq(trainingData, testData) =
      Seq("aclImdb/train/labeledBow-1-index.feat",
        "aclImdb/test/labeledBow-1-index.feat")
        .map(spark.read.format("libsvm").load(_))
    println("Data loaded. Refine data ...")

    val startTime = System.nanoTime

    //refine input data size
    val finalSize = Seq(trainingData, testData).map(_.select("features").first.getAs[SparseVector](0).size).max
    val Seq(resizedTrainingData, resizedTestData) =
      Seq(trainingData, testData)
        .map(_.map( row =>
          (row.getAs[Double](0),
            new SparseVector(finalSize, row.getAs[SparseVector](1).indices, row.getAs[SparseVector](1).values)
          )
        ).toDF("label", "features"))

    println("Data refined. Start regression with logistic regression...")

    logisticRegression(resizedTrainingData, resizedTestData, spark)
    //decisionTree(resizedTrainingData, resizedTestData, spark)
    randomForest(resizedTrainingData, resizedTestData, spark)
    elasticNetLogisticRegression(resizedTrainingData, resizedTestData, spark)

    // Total time
    val finishTime = System.nanoTime
    println("Overall trained + tested time: " + (finishTime - startTime) / 1e9 + " seconds")
    println("Finished")
  }
}
