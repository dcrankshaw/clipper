package ai.clipper

import scala.reflect.runtime.universe
import scala.reflect.runtime.universe.TypeTag
import scala.reflect.api
import scala.io.Source
import java.nio.file.StandardCopyOption.REPLACE_EXISTING
import java.nio.file.StandardOpenOption.CREATE
import java.nio.file.{Files,Paths,Path}
import java.net.URL
import java.net.URLClassLoader
import java.lang.ClassLoader

import org.apache.spark.{SparkContext, SparkConf}

import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.DefaultFormats._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.{read, write}

import org.apache.commons.io.FileUtils

import ai.clipper.container.{MLlibModel,MLlibLoader,MLlibContainer, PipelineModelContainer}
import org.apache.spark.ml.PipelineModel

// implicit val formats = Serialization.formats(NoTypeHints)

case class ClipperContainerConf(
  var className: String,
  var jarNames: String,
  var fromRepl: Boolean = false,
  var replClassDir: Option[String] = None
)


object Clipper {

  // First try to find JAR containing class definition and copy that.
  // If no JAR that contains the class can be found, serialize the class
  // directly.

  // TODO: also try serializing a container instance?
  def deployModel(sc: SparkContext,
                  name: String,
                  version: Int,
                  model: MLlibModel,
                  containerClass: String): Unit = {
    val path = s"/tmp/$name/$version/"
    model.save(sc, path)
    // val containerClass = container.getClass.getName
    // Files.write(Paths.get(s"$path/container.txt"), s"$containerClass\n".getBytes, CREATE)
    val jarPath = Paths.get(getClass.getProtectionDomain.getCodeSource.getLocation.getPath)
    val copiedJarName = "container.jar"
    println(s"JAR path: $jarPath")
    System.out.flush()
    Files.copy(jarPath, Paths.get(s"$path/$copiedJarName"), REPLACE_EXISTING)
    var conf = ClipperContainerConf(containerClass, copiedJarName)
    getReplOutputDir(sc) match {
      case Some(classSourceDir) => {
        println("deployModel called from Spark REPL. Saving classes defined in REPL.")
        conf.fromRepl = true
        val replClassDir = "repl_classes"
        conf.replClassDir = Some(replClassDir)
        val classDestDir = Paths.get(path, replClassDir)
        FileUtils.copyDirectory(Paths.get(classSourceDir).toFile, classDestDir.toFile)
      }
      case None => println("deployModel called from script. No need to save additionally generated classes.")
    }
    implicit val formats = DefaultFormats
    Files.write(Paths.get(path, "container_conf.json"), write(conf).getBytes, CREATE)
  }

  // def getClassLoader(sc: SparkContext, modelDataPath: String): ClassLoader = {
  //
  //
  // }



  def getReplOutputDir(sc: SparkContext) : Option[String] = {
    sc.getConf.getOption("spark.repl.class.outputDir")
  }

  def loadMLLibModel(sc: SparkContext,
                path: String,
                containerClass: String) : MLlibContainer = {
    val model = MLlibLoader.load(sc, path)
    val jarPath = Paths.get(s"$path/container.jar")
    val classLoader = new URLClassLoader(Array(jarPath.toUri.toURL), getClass.getClassLoader)
    val container: MLlibContainer = constructContainer[MLlibContainer](classLoader, containerClass).get
    container.init(sc, model)
    container
  }

  def loadPipelineModel(sc: SparkContext,
                path: String,
                containerClass: String) : PipelineModelContainer = {
    val model = PipelineModel.load(path)
    val jarPath = Paths.get(s"$path/container.jar")
    val classLoader = new URLClassLoader(Array(jarPath.toUri.toURL), getClass.getClassLoader)
    val container: PipelineModelContainer = constructContainer[PipelineModelContainer](classLoader, containerClass).get
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
  def constructContainer[C](classLoader: ClassLoader, containerClass: String) : Option[C] = {
    // val clazz = classLoader.loadClass(containerClass)
    val runtimeMirror: universe.Mirror = universe.runtimeMirror(classLoader)
    val classToLoad = Class.forName(containerClass, true, classLoader)
    // val classSymbol: universe.ClassSymbol = runtimeMirror.classSymbol(Class.forName(containerClass))
    val classSymbol: universe.ClassSymbol = runtimeMirror.classSymbol(classToLoad)
    val classMirror: universe.ClassMirror = runtimeMirror.reflectClass(classSymbol)
    val constructorMirror = classMirror.reflectConstructor(selectConstructor(classSymbol))
    try {
      return Some(constructorMirror().asInstanceOf[C])
    } catch {
      case wrongClass: ClassCastException => println(s"Could not cast provided class: $wrongClass")
      case e: Throwable => println(s"Error loading constructor: $e")
    }
    None
  }

  // def constructPipelineContainer(classLoader: ClassLoader, containerClass: String) : Option[MLlibContainer] = {
  //   val clazz = classLoader.loadClass(containerClass)
  //   val runtimeMirror: universe.Mirror = universe.runtimeMirror(classLoader)
  //   val classToLoad = Class.forName(containerClass, true, classLoader)
  //   // val classSymbol: universe.ClassSymbol = runtimeMirror.classSymbol(Class.forName(containerClass))
  //   val classSymbol: universe.ClassSymbol = runtimeMirror.classSymbol(classToLoad)
  //   val classMirror: universe.ClassMirror = runtimeMirror.reflectClass(classSymbol)
  //   val constructorMirror = classMirror.reflectConstructor(selectConstructor(classSymbol))
  //   try {
  //     return Some(constructorMirror().asInstanceOf[MLlibContainer])
  //   } catch {
  //     case wrongClass: ClassCastException => println(s"Could not cast provided class to MLlibContainer: $wrongClass")
  //     case e: Throwable => println(s"Error loading constructor: $e")
  //   }
  //   None
  // }
}
