package clipper
import scala.io.Source

object Serve {

  def main(args: Array[String]) : Unit = {

    val model_path = sys.env("CLIPPER_MODEL_DATA")
    val model_name = sys.env("CLIPPER_MODEL_NAME")
    val model_version = sys.env("CLIPPER_MODEL_VERSION").toInt


    val conf = new SparkConf()
      .setAppName("ClipperMLLibContainer")
      .setMaster("local")
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
    val container: Container =
      Clipper.loadModel(sc, model_path)


    val labelAndPreds = testData.collect().map { point =>
      val prediction = container.predict(point.features)
      (point.label, prediction)
    }
    val numWrong = labelAndPreds.filter(r => r._1 != r._2).size
    println(s"Test Error $numWrong")
  }

  def getContainerClass(path: String) : String = {
    val filename = s"$path/container.txt"
    Source.fromFile(filename).getLines.take(1)(0)
  }

}

