
package clipper

// import scala.reflect.runtime.universe._
import scala.reflect.runtime.universe
import scala.reflect.runtime.universe.TypeTag
import scala.reflect.api
import java.io._
import scala.io.Source


import org.json4s._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.Saveable
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.clustering.BisectingKMeansModel


// sealed abstract class MLlibModel {
//   def predict(features: Vector): Double
//   def save(sc: SparkContext, path: String): Unit
// }
//
// // LogisticRegressionModel
// case class MLlibLogisticRegressionModel(model: LogisticRegressionModel) extends MLlibModel {
//   override def predict(features: Vector): Double  = {
//     model.predict(features)
//   }
//
//   override def save(sc: SparkContext, path: String): Unit = {
//     model.save(sc, path)
//   }
// }
//
// // NaiveBayesModel
// case class MLlibNaiveBayesModel(model: NaiveBayesModel) extends MLlibModel {
//   override def predict(features: Vector): Double  = {
//     model.predict(features)
//   }
//
//   override def save(sc: SparkContext, path: String): Unit = {
//     model.save(sc, path)
//   }
// }
//
//
// // SVMModel
// case class MLlibSVMModel(model: SVMModel) extends MLlibModel {
//   override def predict(features: Vector): Double  = {
//     model.predict(features)
//   }
//
//   override def save(sc: SparkContext, path: String): Unit = {
//     model.save(sc, path)
//   }
// }
//
//
// // BisectingKMeansModel
// case class MLlibBisectingKMeansModel(model: BisectingKMeansModel) extends MLlibModel {
//   override def predict(features: Vector): Double  = {
//     model.predict(features)
//   }
//
//   override def save(sc: SparkContext, path: String): Unit = {
//     model.save(sc, path)
//   }
// }




class LogisticRegressionContainer extends Container {

  // var model: Option[MLlibLogisticRegressionModel] = None
  var model: Option[MLlibModel] = None

  override def init(sc: SparkContext, m: MLlibModel) {
    // model = Some(model.asInstanceOf[MLlibLogisticRegressionModel)
    println("Initializing container")
    model = Some(m)
  }

  override def predict(x: Vector): Double = {
    println("making prediction")
    val m = model.get
    m.predict(x)
    // xs.map(m.predict(_))
  }
}


// object Clipper {
//
//   // TODO: also try serializing a container instance?
//   def deployModel(sc: SparkContext,
//                   name: String,
//                   version: Int,
//                   model: MLlibModel,
//                   containerClass: String): Unit = {
//     val path = s"/tmp/$name/$version/"
//     model.save(sc, path)
//     // val containerClass = container.getClass.getName
//     val pw = new PrintWriter(new File(s"$path/container.txt" ))
//     pw.write(containerClass)
//     pw.close
//
//   }
//
//   def loadModel(sc: SparkContext,
//                 path: String,
//                 containerClass: String) : Container = {
//     val model = MLlibLoader.load(sc, path)
//     val container: Container = constructContainer(containerClass).get
//     container.init(sc, model)
//     container
//   }
//
//   private def selectConstructor(symbol: universe.Symbol) = {
//       val constructors = symbol.typeSignature.members.filter(_.isConstructor).toList
//       if (constructors.length > 1) println(
//              s"""Warning: $symbol has several constructors, arbitrarily picking the first one: 
//                 |         ${constructors.mkString("\n         ")}""".stripMargin)
//       constructors.head.asMethod
//     }
//
//   // adapted from http://stackoverflow.com/q/34227984/814642
//   def constructContainer(containerClass: String) : Option[Container] = {
//     val runtimeMirror: universe.Mirror = universe.runtimeMirror(getClass.getClassLoader)
//     val classSymbol: universe.ClassSymbol = runtimeMirror.classSymbol(Class.forName(containerClass))
//     val classMirror: universe.ClassMirror = runtimeMirror.reflectClass(classSymbol)
//     val constructorMirror = classMirror.reflectConstructor(selectConstructor(classSymbol))
//     try {
//       return Some(constructorMirror().asInstanceOf[Container])
//     } catch {
//       case wrongClass: ClassCastException => println(s"Could not cast provided class to Container: $wrongClass")
//       case e: Throwable => println(s"Error loading constructor: $e")
//     }
//     None
//   }
// }

object Train {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ClipperTest").setMaster("local[2]")
    val sc = new SparkContext(conf)
    sc.parallelize(Seq("")).foreachPartition(x => {
        import org.apache.log4j.{LogManager, Level}
        import org.apache.commons.logging.LogFactory
        LogManager.getRootLogger().setLevel(Level.WARN)
        val log = LogFactory.getLog("EXECUTOR-LOG:")
        log.warn("START EXECUTOR WARN LOG LEVEL")
      })

    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(
      sc,
      "/Users/crankshaw/model-serving/spark_serialization_project/spark_binary/data/mllib/sample_libsvm_data.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    val numClasses = 2
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    // val categoricalFeaturesInfo = Map[Int, Int]()
    // val impurity = "gini"
    // val maxDepth = 5
    // val maxBins = 32

    val model = MLlibLogisticRegressionModel(new LogisticRegressionWithLBFGS().setNumClasses(numClasses).run(trainingData))

    // val model = MLlibLogisticRegressionModel(LogisticRegressionModelWithSGD.train(trainingData,
    //                                          numClasses,
    //                                          categoricalFeaturesInfo,
    //                                          impurity,
    //                                          maxDepth,
    //                                          maxBins))



    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val numWrong = labelAndPreds.filter(r => r._1 != r._2).count()
    println(s"Num wrong: $numWrong")
    println("Learned logistic regression model:\n" + model.toString)

    Clipper.deployModel(sc, "test", 1, model, "clipper.sandbox.LogisticRegressionContainer")
    sc.stop()
  }

  // def serveModel() : Unit = {
  //   val conf = new SparkConf().setAppName("ClipperTest").setMaster("local[2]")
  //   val sc = new SparkContext(conf)
  //   sc.parallelize(Seq("")).foreachPartition(x => {
  //       import org.apache.log4j.{LogManager, Level}
  //       import org.apache.commons.logging.LogFactory
  //       LogManager.getRootLogger().setLevel(Level.WARN)
  //       val log = LogFactory.getLog("EXECUTOR-LOG:")
  //       log.warn("START EXECUTOR WARN LOG LEVEL")
  //     })
  //
  //   // Load and parse the data file.
  //   val data = MLUtils.loadLibSVMFile(
  //     sc,
  //     "/Users/crankshaw/model-serving/spark_serialization_project/spark_binary/data/mllib/sample_libsvm_data.txt")
  //   // Split the data into training and test sets (30% held out for testing)
  //   val splits = data.randomSplit(Array(0.7, 0.3))
  //   val (trainingData, testData) = (splits(0), splits(1))
  //   val path = "/tmp/test/1"
  //   val container: Container = Clipper.loadModel(sc, path, "clipper.sandbox.LogisticRegressionContainer")
  //
  //
  //   val labelAndPreds = testData.collect().map { point =>
  //     val prediction = container.predict(point.features)
  //     (point.label, prediction)
  //   }
  //   val numWrong = labelAndPreds.filter(r => r._1 != r._2).size
  //   println(s"Test Error $numWrong")
  // }
}


// GaussianMixtureModel
// KMeansModel
// LDAModel
// PowerIterationClusteringModel
// ChiSqSelectorModel
// Word2VecModel
// FPGrowthModel
// PrefixSpanModel
// MatrixFactorizationModel
// IsotonicRegressionModel
// LassoModel
// LinearRegressionModel
// RidgeRegressionModel
// DecisionTreeModel
// RandomForestModel
// GradientBoostedTreesModel
