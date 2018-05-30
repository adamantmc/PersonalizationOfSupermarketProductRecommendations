import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.SparkConf
import DataReader.read_data
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.mutable

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

        val (groceries_df, categories_df) = read_data(
            "data/groceries.csv",
            "data/products-categorized.csv",
            spark
        )

        val groceries_cast = cast_groceries_to_classes(groceries_df, categories_df, spark)
        val customer_vectors_rdd = vectorize_customers(groceries_cast, categories_df, "customer_vectors", spark)
    }

    def cast_groceries_to_classes(groceries: DataFrame,
                                  categories: DataFrame,
                                  spark: SparkSession,
                                  subclass_groceries_output_col: String = "subclass_groceries",
                                  class_groceries_output_col: String = "class_groceries"): DataFrame = {

        import spark.sqlContext.implicits._

        val udf_product_in_array = udf(
            (groceries_arr: mutable.WrappedArray[String], product: String) => groceries_arr.contains(product)
        )

        /*
            We join two rows of the groceries and categories dataframes if the product in the 'categories' row appears
            in the basket of the 'groceries' row. Then we group by the basket_id, aggregate subclasses and classes
            into lists, and re-join with the groceries dataframe.
         */
        val groceries_joined = groceries
            .join(categories, udf_product_in_array($"groceries", categories.col("product")))
            .groupBy("basket_id")
            .agg(collect_list("subclass"), collect_list("class"))
            .toDF("basket_id", subclass_groceries_output_col, class_groceries_output_col)
            .join(groceries, "basket_id")

        groceries_joined
    }

    def vectorize_customers(groceries: DataFrame, categories: DataFrame, outputCol: String, spark: SparkSession): RDD[(Int, Array[Double])] = {
        import spark.sqlContext.implicits._

        /*
            We must vectorize each product subclass basket, and add them all together to get the total spending
            of a customer per subclass. We will use a CountVectorizer, fitted in the subclasses.
         */

        val subclassCountVectorizer = new CountVectorizer()
            .setInputCol("subclass_groceries")
            .fit(groceries)

        val groceries_subclass_vectors = subclassCountVectorizer
            .setInputCol("subclass_groceries")
            .setOutputCol("subclass_vector")
            .transform(groceries)

        groceries_subclass_vectors.show(10, false)

        /*
            Now we have the vectors representing the spendings of each customer per subclass. Next step is
            to normalize them, first by the total spending of the customer and then by the spending of other
            customers per subclass.
         */

        // Keep customer vectors as RDD - still not aggergated by customer id
        val customer_id_index = groceries_subclass_vectors.columns.indexOf("customer_id")
        val customer_vector_index = groceries_subclass_vectors.columns.indexOf("subclass_vector")

        val customer_vectors_rdd = groceries_subclass_vectors.rdd.map(x => (x.getInt(customer_id_index), x.getAs[SparseVector](customer_vector_index)))

        val product_subclasses = subclassCountVectorizer.vocabulary.length

        val add_op = (a: Double, b: Double) => a + b
        val div_op = (a: Double, b: Double) => a / b

        val summed_vectors_per_customer = customer_vectors_rdd.map(x => (x._1, x._2.toArray))
            .reduceByKey((arr1, arr2) => array_op_element_wise(arr1, arr2, add_op))
            .map(x => (x._1, array_div_scalar(x._2, array_sum(x._2))))

        /*
            The customer vectors are now normalized by their total spending -
            we must now normalize each subclass spending per customer with the average total spending per subclass.
        */

        val number_of_customers = summed_vectors_per_customer.count()

        // Normalize total spending vector by number of customers
        val total_spending_vector_normalized = array_div_scalar(
            summed_vectors_per_customer.values.reduce((arr1, arr2) => array_op_element_wise(arr1, arr2, add_op)),
            number_of_customers
        )

        // Normalize customer vectors by total spending in each subclass
        val final_customer_vectors = summed_vectors_per_customer
            .map(x => (x._1, array_op_element_wise(x._2, total_spending_vector_normalized, div_op)))

        final_customer_vectors
    }

    def array_op_element_wise(arr1: Array[Double], arr2: Array[Double], op: (Double, Double) => Double): Array[Double] = {
        var result = new Array[Double](arr1.length)

        var i = 0
        while (i < arr1.length) {
            result(i) = op(arr1(i), arr2(i))
            i += 1
        }

        result
    }

    def array_sum(arr: Array[Double]): Double = {
        var sum = 0.0
        arr.foreach(v => sum += v)
        sum
    }

    def array_div_scalar(arr: Array[Double], value: Double): Array[Double] = {
        arr.map(v => v / value)
    }
}