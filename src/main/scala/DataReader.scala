import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer}

object DataReader {

    def read_groceries(csv_file: String, spark: SparkSession): DataFrame = {
        import spark.sqlContext.implicits._

        val groceries_rdd = spark.sparkContext.textFile(csv_file)
            .map(str => str.split(","))
            .zipWithIndex()
            .map(x => (x._2, x._1))

        val groceries_df = groceries_rdd.toDF("id", "groceries")

        val countVectorizer = new CountVectorizer()
            .setInputCol("groceries")
            .setOutputCol("groceries_vectors")
            .fit(groceries_df)

        countVectorizer.transform(groceries_df)
    }



}