package clipper.container

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

sealed abstract class MLlibModel {
  def predict(features: Vector): Double
  def save(sc: SparkContext, path: String): Unit
}

// LogisticRegressionModel
case class MLlibLogisticRegressionModel(model: LogisticRegressionModel) extends MLlibModel {
  override def predict(features: Vector): Double  = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// NaiveBayesModel
case class MLlibNaiveBayesModel(model: NaiveBayesModel) extends MLlibModel {
  override def predict(features: Vector): Double  = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}


// SVMModel
case class MLlibSVMModel(model: SVMModel) extends MLlibModel {
  override def predict(features: Vector): Double  = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}


// BisectingKMeansModel
case class MLlibBisectingKMeansModel(model: BisectingKMeansModel) extends MLlibModel {
  override def predict(features: Vector): Double  = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

abstract class Container {

  def init(sc: SparkContext, model: MLlibModel): Unit

  def predict(xs: Vector) : Double

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
