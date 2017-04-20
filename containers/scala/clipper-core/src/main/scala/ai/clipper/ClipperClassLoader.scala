package ai.clipper

import java.io.{ByteArrayOutputStream, FileNotFoundException, FilterInputStream, InputStream, IOException}
import org.apache.xbean.asm5._
import org.apache.xbean.asm5.Opcodes._

import java.net.{HttpURLConnection, URI, URL, URLEncoder}
import scala.io.Source
import java.nio.file.StandardCopyOption.REPLACE_EXISTING
import java.nio.file.StandardOpenOption.CREATE
import java.nio.file.{Files,Paths,Path}

/**
 * A class loader which makes some protected methods in ClassLoader accessible.
 */
class ParentClassLoader(parent: ClassLoader) extends ClassLoader(parent) {

  override def findClass(name: String): Class[_] = {
    super.findClass(name)
  }

  override def loadClass(name: String): Class[_] = {
    super.loadClass(name)
  }

  override def loadClass(name: String, resolve: Boolean): Class[_] = {
    super.loadClass(name, resolve)
  }

}


/**
 * A ClassLoader that first tries to load classes from the parent ClassLoader and
 * otherwise will load REPL defined classes stored in the directory specified
 * by replClassUri
 *
 * @param classRootDir The directory corresponding to Spark's spark.repl.class.outputDir
 * option. The directory will be copied into the model container, and this option should
 * be the path of the copied directory.
 */
class ContainerClassLoader(parent: ClassLoader, classRootDir: String) extends ClassLoader {


  // val directory = uri.getPath
  val parentLoader = new ParentClassLoader(parent)

  override def getResource(name: String): URL = {
    parentLoader.getResource(name)
  }

  override def getResources(name: String): java.util.Enumeration[URL] = {
    parentLoader.getResources(name)
  }

  override def findClass(name: String): Class[_] = {
    try {
      // First try to load with parent class loader
      parentLoader.loadClass(name)
    } catch {
      case e: ClassNotFoundException =>
        val classOption = findReplClass(name)
        classOption match {
          case None => throw new ClassNotFoundException(name, e)
          case Some(a) => a
        }
    }
  }

  def findReplClass(name: String) : Option[Class[_]] = {
    val pathInDirectory = name.replace('.', '/') + ".class"
    var inputStream: InputStream = null
    try {
      inputStream = getClassFileInputStream(pathInDirectory)
      val bytes = readAndTransformClass(name, inputStream)
      Some(defineClass(name, bytes, 0, bytes.length))
    } catch {
      case e: ClassNotFoundException =>
        // We did not find the class
        println(s"Did not load class $name from REPL class directory", e)
        None
      case e: Exception =>
        // Something bad happened while checking if the class exists
        println(s"Failed to check existence of class $name in REPL class directory", e)
        None
    } finally {
      if (inputStream != null) {
        try {
          inputStream.close()
        } catch {
          case e: Exception =>
            println("Exception while closing inputStream", e)
        }
      }
    }
  }


  private def getClassFileInputStream(pathInDirectory: String): InputStream = {
    val path = Paths.get(classRootDir, pathInDirectory)
    try {
      Files.newInputStream(path)
    } catch {
      case _: FileNotFoundException =>
        throw new ClassNotFoundException(s"Class file not found at path $path")
    }
  }

  def readAndTransformClass(name: String, in: InputStream): Array[Byte] = {
    if (name.startsWith("line") && name.endsWith("$iw$")) {
      // Class seems to be an interpreter "wrapper" object storing a val or var.
      // Replace its constructor with a dummy one that does not run the
      // initialization code placed there by the REPL. The val or var will
      // be initialized later through reflection when it is used in a task.
      val cr = new ClassReader(in)
      val cw = new ClassWriter(
        ClassWriter.COMPUTE_FRAMES + ClassWriter.COMPUTE_MAXS)
      val cleaner = new ConstructorCleaner(name, cw)
      cr.accept(cleaner, 0)
      return cw.toByteArray
    } else {
      // Pass the class through unmodified
      val bos = new ByteArrayOutputStream
      val bytes = new Array[Byte](4096)
      var done = false
      while (!done) {
        val num = in.read(bytes)
        if (num >= 0) {
          bos.write(bytes, 0, num)
        } else {
          done = true
        }
      }
      return bos.toByteArray
    }
  }
}

// Copied from org.apache.spark.repl.ConstructorCleaner
class ConstructorCleaner(className: String, cv: ClassVisitor)
extends ClassVisitor(ASM5, cv) {
  override def visitMethod(access: Int, name: String, desc: String,
      sig: String, exceptions: Array[String]): MethodVisitor = {
    val mv = cv.visitMethod(access, name, desc, sig, exceptions)
    if (name == "<init>" && (access & ACC_STATIC) == 0) {
      // This is the constructor, time to clean it; just output some new
      // instructions to mv that create the object and set the static MODULE$
      // field in the class to point to it, but do nothing otherwise.
      mv.visitCode()
      mv.visitVarInsn(ALOAD, 0) // load this
      mv.visitMethodInsn(INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false)
      mv.visitVarInsn(ALOAD, 0) // load this
      // val classType = className.replace('.', '/')
      // mv.visitFieldInsn(PUTSTATIC, classType, "MODULE$", "L" + classType + ";")
      mv.visitInsn(RETURN)
      mv.visitMaxs(-1, -1) // stack size and local vars will be auto-computed
      mv.visitEnd()
      return null
    } else {
      return mv
    }
  }
}
