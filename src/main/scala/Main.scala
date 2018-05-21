import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.SparkConf
import DataReader.read_groceries

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

        val groceries_df = read_groceries("data/groceries.csv", spark)
    }
}