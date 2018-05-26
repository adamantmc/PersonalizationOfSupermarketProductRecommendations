import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.CountVectorizer

object DataReader {

    def read_data(groceries_file: String, product_categories_file: String, spark: SparkSession): DataFrame = {
        import spark.sqlContext.implicits._

        val groceries_rdd = spark.sparkContext.textFile(groceries_file)
            .map(str => str.split(","))
            .zipWithIndex()
            .map(x => (x._2, x._1))

        val groceries_df = groceries_rdd.toDF("id", "groceries")

        val countVectorizer = new CountVectorizer()
            .setInputCol("groceries")
            .setOutputCol("groceries_vectors")
            .fit(groceries_df)

        val transformed_groceries_df = countVectorizer.transform(groceries_df)

        val product_categories_rdd = spark.sparkContext.textFile(product_categories_file)
            .map(str => str.split(","))
            .map(arr => (arr(0), arr(1).split("/").padTo(2, "")))

        val product_categories_df =
            product_categories_rdd.toDF("product", "categories")
                .withColumn("class", $"categories".getItem(0))
                .withColumn("subclass", $"categories".getItem(1))

        product_categories_df.show(10, false)

        transformed_groceries_df
    }


}