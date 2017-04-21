package ai.clipper

import java.net.URLClassLoader
import java.nio.file.StandardCopyOption.REPLACE_EXISTING
import java.nio.file.StandardOpenOption.CREATE
import java.nio.file.{Files, Paths}

import ai.clipper.container._
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.apache.spark.ml.PipelineModel
import org.json4s._
import org.json4s.jackson.Serialization.{read, write}

import scala.reflect.runtime.universe

sealed trait ModelType extends Serializable

case object MLlibModelType extends ModelType

case object PipelinesModelType extends ModelType

case class ClipperContainerConf(var className: String,
                                var jarName: String,
                                var modelType: ModelType,
                                var fromRepl: Boolean = false,
                                var replClassDir: Option[String] = None)

object Clipper {

  val CLIPPER_CONF_FILENAME: String = "clipper_conf.json"
  val CONTAINER_JAR_FILE: String = "container_source.jar"
  val MODEL_DIRECTORY: String = "model"
  val REPL_CLASS_DIR: String = "repl_classes"


  def deployModel[M](sc: SparkContext,
                          name: String,
                          version: Int,
                          model: M,
                          containerClass: Class[_]): Unit = {
    val basePath = Paths.get("/tmp", name, version.toString).toString
    val modelPath = Paths.get(basePath, MODEL_DIRECTORY).toString
    val modelType = model match {
      case m: MLlibModel => {
        m.save(sc, modelPath)
        // Because I'm not sure how to do it in the type system, check that
        // the container is of the right type
        try {
          containerClass.newInstance.asInstanceOf[MLlibContainer]
        } catch {
          case e: ClassCastException => {
            throw new IllegalArgumentException(
              "Error: Container must be a subclass of MLlibContainer")
          }
        }
        MLlibModelType
      }
      case p: PipelineModel => {
        p.save(modelPath)
        // Because I'm not sure how to do it in the type system, check that
        // the container is of the right type
        try {
          containerClass.newInstance.asInstanceOf[PipelineModelContainer]
        } catch {
          case e: ClassCastException => {
            throw new IllegalArgumentException(
              "Error: Container must be a subclass of PipelineModelContainer")
          }
        }
        PipelinesModelType
      }
      case _ =>
        throw new IllegalArgumentException(
          s"Illegal model type: ${model.getClass.getName}")
    }
    // val containerClass = container.getClass.getName
    // Files.write(Paths.get(s"$path/container.txt"), s"$containerClass\n".getBytes, CREATE)
    val jarPath = Paths.get(
      containerClass.getProtectionDomain.getCodeSource.getLocation.getPath)
    val copiedJarName = CONTAINER_JAR_FILE
    println(s"JAR path: $jarPath")
    System.out.flush()
    Files.copy(jarPath, Paths.get(basePath, copiedJarName), REPLACE_EXISTING)
    val conf =
      ClipperContainerConf(containerClass.getName, copiedJarName, modelType)
    getReplOutputDir(sc) match {
      case Some(classSourceDir) => {
        println(
          "deployModel called from Spark REPL. Saving classes defined in REPL.")
        conf.fromRepl = true
        conf.replClassDir = Some(REPL_CLASS_DIR)
        val classDestDir = Paths.get(basePath, REPL_CLASS_DIR)
        FileUtils.copyDirectory(Paths.get(classSourceDir).toFile,
                                classDestDir.toFile)
      }
      case None =>
        println(
          "deployModel called from script. No need to save additionally generated classes.")
    }
    implicit val formats = DefaultFormats
    Files.write(Paths.get(basePath, CLIPPER_CONF_FILENAME),
                write(conf).getBytes,
                CREATE)
  }

  private def getReplOutputDir(sc: SparkContext): Option[String] = {
    sc.getConf.getOption("spark.repl.class.outputDir")
  }

  def loadModel(sc: SparkContext, modelDataPath: String): Container = {
    val confString = Files
      .readAllBytes(Paths.get(modelDataPath, CLIPPER_CONF_FILENAME))
      .toString
    val conf = read[ClipperContainerConf](confString)
    val classLoader = getClassLoader(modelDataPath, conf)
    conf.modelType match {
      case MLlibModelType => {
        val model = MLlibLoader.load(sc, modelDataPath)
        val container =
          constructContainer[MLlibContainer](classLoader, conf.className).get
        container.init(sc, model)
        container.asInstanceOf[Container]
      }
      case PipelinesModelType => {
        val model = PipelineModel.load(modelDataPath)
        val container = constructContainer[PipelineModelContainer](
          classLoader,
          conf.className).get
        container.init(sc, model)
        container.asInstanceOf[Container]
      }
    }
  }

  private def getClassLoader(path: String, conf: ClipperContainerConf): ClassLoader = {
    val parentLoader = new URLClassLoader(
      Array(Paths.get(path, conf.jarName).toUri.toURL),
      getClass.getClassLoader)
    if (conf.fromRepl) {
      println("Creating Clipper REPL ClassLoader")
      new ClipperClassLoader(parentLoader,
                             Paths.get(path, conf.replClassDir.get).toString)
    } else {
      parentLoader
    }
  }

  // adapted from http://stackoverflow.com/q/34227984/814642
  private def constructContainer[C](classLoader: ClassLoader,
                            containerClass: String): Option[C] = {
    // val clazz = classLoader.loadClass(containerClass)
    val runtimeMirror: universe.Mirror = universe.runtimeMirror(classLoader)
    val classToLoad = Class.forName(containerClass, true, classLoader)
    // val classSymbol: universe.ClassSymbol = runtimeMirror.classSymbol(Class.forName(containerClass))
    val classSymbol: universe.ClassSymbol =
      runtimeMirror.classSymbol(classToLoad)
    val classMirror: universe.ClassMirror =
      runtimeMirror.reflectClass(classSymbol)
    val constructorMirror =
      classMirror.reflectConstructor(selectConstructor(classSymbol))
    try {
      return Some(constructorMirror().asInstanceOf[C])
    } catch {
      case wrongClass: ClassCastException =>
        println(s"Could not cast provided class: $wrongClass")
      case e: Throwable => println(s"Error loading constructor: $e")
    }
    None
  }

  private def selectConstructor(symbol: universe.Symbol) = {
    val constructors =
      symbol.typeSignature.members.filter(_.isConstructor).toList
    if (constructors.length > 1)
      println(
        s"""Warning: $symbol has several constructors, arbitrarily picking the first one:
         |         ${constructors.mkString("\n         ")}""".stripMargin)
    constructors.head.asMethod
  }
}
