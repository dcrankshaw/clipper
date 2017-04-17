package clipper

import scala.reflect.runtime.universe
import scala.reflect.runtime.universe.TypeTag
import scala.reflect.api
import java.io._
import scala.io.Source

import org.json4s._
import org.json4s.jackson.JsonMethods._

object MLlibLoader {
  def metadataPath(path: String): String = s"$path/metadata"

  def getModelClassName(sc: SparkContext, path: String) = {
    val str = sc.textFile(metadataPath(path)).take(1)(0)
    println(s"JSON STRING: $str")
    val json = parse(str)
    val JString(className) = (json \ "class")
    className
  }

  def load(sc: SparkContext, path: String): MLlibModel = {
    val className = getModelClassName(sc, path)
    // Reflection Code
    val mirror = universe.runtimeMirror(getClass.getClassLoader)
    val modelModule = mirror.staticModule(className)
    val anyInst = mirror.reflectModule(modelModule).instance
    val loader = anyInst.asInstanceOf[org.apache.spark.mllib.util.Loader[_]]
    val model = loader.load(sc, path) match {
      case model: LogisticRegressionModel => MLlibLogisticRegressionModel(model)
      case model: NaiveBayesModel => MLlibNaiveBayesModel(model)
      case model: SVMModel => MLlibSVMModel(model)
      case model: BisectingKMeansModel => MLlibBisectingKMeansModel(model)
    }
    model
  }
}

object Clipper {

  // TODO: also try serializing a container instance?
  def deployModel(sc: SparkContext,
                  name: String,
                  version: Int,
                  model: MLlibModel,
                  containerClass: String): Unit = {
    val path = s"/tmp/$name/$version/"
    model.save(sc, path)
    // val containerClass = container.getClass.getName
    val pw = new PrintWriter(new File(s"$path/container.txt" ))
    pw.write(containerClass)
    pw.close

  }

  def loadModel(sc: SparkContext,
                path: String,
                containerClass: String) : Container = {
    val model = MLlibLoader.load(sc, path)
    val container: Container = constructContainer(containerClass).get
    container.init(sc, model)
    container
  }

  private def selectConstructor(symbol: universe.Symbol) = {
      val constructors = symbol.typeSignature.members.filter(_.isConstructor).toList
      if (constructors.length > 1) println(
             s"""Warning: $symbol has several constructors, arbitrarily picking the first one: 
                |         ${constructors.mkString("\n         ")}""".stripMargin)
      constructors.head.asMethod
    }

  // adapted from http://stackoverflow.com/q/34227984/814642
  def constructContainer(containerClass: String) : Option[Container] = {
    val runtimeMirror: universe.Mirror = universe.runtimeMirror(getClass.getClassLoader)
    val classSymbol: universe.ClassSymbol = runtimeMirror.classSymbol(Class.forName(containerClass))
    val classMirror: universe.ClassMirror = runtimeMirror.reflectClass(classSymbol)
    val constructorMirror = classMirror.reflectConstructor(selectConstructor(classSymbol))
    try {
      return Some(constructorMirror().asInstanceOf[Container])
    } catch {
      case wrongClass: ClassCastException => println(s"Could not cast provided class to Container: $wrongClass")
      case e: Throwable => println(s"Error loading constructor: $e")
    }
    None
  }
}
