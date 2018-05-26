import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{SparkSession}
import org.apache.spark.SparkConf
import DataReader.read_data

object Main {
    def main(args: Array[String]): Unit = {
        val conf = new SparkConf()
            .setMaster("local[4]")
            .set("spark.executor.memory", "1g")

        val spark = SparkSession
            .builder()
            .appName("Personalization of supermarket product recommendations")
            .config(conf)
            .getOrCreate()

        Logger.getLogger("org").setLevel(Level.WARN)

        val groceries_df = read_data(
            "data/groceries.csv",
            "data/products-categorized.csv",
            spark
        )
    }
}