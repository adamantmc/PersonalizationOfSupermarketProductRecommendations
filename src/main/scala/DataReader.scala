import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.CountVectorizer
import scala.util.Random

object DataReader {

    def read_data(groceries_file: String, product_categories_file: String, spark: SparkSession): (DataFrame, DataFrame) = {
        import spark.sqlContext.implicits._

        val random = Random
        val number_of_customers = 1000

        val groceries_rdd = spark.sparkContext.textFile(groceries_file)
            .zipWithIndex()
            .map(x => (random.nextInt(number_of_customers), x._2, x._1.split(",").map(x => x.trim())))

        val groceries_df = groceries_rdd.toDF("customer_id", "basket_id", "groceries")

        val product_categories_rdd = spark.sparkContext.textFile(product_categories_file)
            .map(str => str.split(","))
            .map(arr => (arr(0).trim(), arr(1).split("/").padTo(2, "").map(x => x.trim())))

        val product_categories_df =
            product_categories_rdd.toDF("product", "categories")
                .withColumn("class", $"categories".getItem(0))
                .withColumn("subclass", $"categories".getItem(1))
                .filter($"class" =!= "" && $"class" =!= "?" && $"subclass" =!= "" && $"subclass" =!= "?")

        (groceries_df, product_categories_df)
    }


}