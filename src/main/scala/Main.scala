import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.SparkConf
import DataReader.read_data
import breeze.linalg.Matrix
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.rdd.RDD

import scala.collection.mutable

object Main {

    val udf_product_in_array: UserDefinedFunction = udf(
        (groceries_arr: mutable.WrappedArray[String], product: String) => groceries_arr.contains(product)
    )

    val udf_str_to_array: UserDefinedFunction = udf(
        (str: String) => Array[String](str)
    )

    val udf_arr_to_str: UserDefinedFunction = udf(
        (arr: mutable.WrappedArray[String]) => arr(0)
    )

    def main(args: Array[String]): Unit = {
        val conf = new SparkConf()
          .setMaster("local[4]")
          .set("spark.executor.memory", "1g")

        val spark = SparkSession
          .builder()
          .appName("Personalization of supermarket product recommendations")
          .config(conf)
          .getOrCreate()

        import spark.sqlContext.implicits._

        Logger.getLogger("org").setLevel(Level.WARN)

        val (groceries_df, categories_df) = read_data(
            "data/groceries.csv",
            "data/products-categorized.csv",
            spark
        )

        /*
            Cast product baskets to subclass and class baskets
            Format: basket_id: Int,
                    subclass_groceries: Array[String], class_groceries: Array[String],
                    customer_id: Int, groceries: Array[String]
        */
        val groceries_cast = cast_groceries_to_classes(groceries_df, categories_df, spark)

        val subclassCountVectorizer = new CountVectorizer()
            .setInputCol("subclass_groceries")
            .fit(groceries_cast)

        val class_rules =
            association_rules(groceries_cast, "class_groceries", 0.1, 0.6, spark)

        val subclass_rules =
            association_rules(groceries_cast, "subclass_groceries", 0.05, 0.4, spark)

        val customer_vectors_rdd = vectorize_customers(groceries_cast, subclassCountVectorizer, "customer_vectors", spark)
        val product_vectors_rdd = vectorize_products(
            categories_df, subclassCountVectorizer.vocabulary,
            class_rules, subclass_rules,
            "product_vectors", spark
        )

        val recommendations = customer_vectors_rdd.cartesian(product_vectors_rdd)
            .map(x => (x._1._1, (x._2._1, cosineSimilarity(x._1._2, x._2._2)))) // map to (customer, (product, similarity))
            .groupByKey()
            .map(x => (x._1, x._2.toSeq.sortBy(x => -x._2).take(10)))

        recommendations.toDF().show(truncate = false)
    }

    /*
  * Cosine similarity of two vectors of same dimensionality
  * cos(x, y) = (x dot y) / (||X||*||Y||)
  */
    def cosineSimilarity(x: Array[Double], y: Array[Double]): Double = {
        var dot_product = 0.0
        var x_mag = 0.0
        var y_mag = 0.0

        for((a, b) <- x zip y) {
            dot_product += a * b
            x_mag += a * a
            y_mag += b * b
        }

        x_mag = math.sqrt(x_mag)
        y_mag = math.sqrt(y_mag)

       dot_product / (x_mag * y_mag)
    }

    def cast_groceries_to_classes(groceries: DataFrame,
                                  categories: DataFrame,
                                  spark: SparkSession,
                                  subclass_groceries_output_col: String = "subclass_groceries",
                                  class_groceries_output_col: String = "class_groceries"): DataFrame = {
        import spark.sqlContext.implicits._

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

    def association_rules(groceries: DataFrame,
                          inputCol: String,
                          minSupport: Double, minConf: Double,
                          spark: SparkSession) = {
        import spark.sqlContext.implicits._

        /*
            Input column contains baskets which might have duplicate entries. We need to get
            rid of them.
         */
        val udf_discretize_array = udf((groceries_arr: mutable.WrappedArray[String]) => groceries_arr.distinct)

        val fpgrowth = new FPGrowth()
            .setItemsCol("baskets_distinct")
            .setMinSupport(minSupport)
            .setMinConfidence(minConf)
            .fit(groceries.withColumn("baskets_distinct", udf_discretize_array(groceries.col(inputCol))))

        /*
            We only care about rules with one product on the Left Hand Side.
            Filter by lhs length, then cast columns from arrays to strings. Finally,
            aggregate the right hand side column for each left hand side.
         */
        fpgrowth.associationRules
            .filter(size($"antecedent") === 1)
            .withColumn("lhs", udf_arr_to_str($"antecedent"))
            .withColumn("rhs", udf_arr_to_str($"consequent"))
            .groupBy($"lhs")
            .agg(collect_list($"rhs") as "rhs")
            .drop("antecedent")
            .drop("consequent")
    }

    def vectorize_products(categories: DataFrame, subclasses: Array[String],
                           class_rules: DataFrame, subclass_rules: DataFrame,
                           outputCol: String, spark: SparkSession): RDD[(String, Array[Double])] = {
        import spark.sqlContext.implicits._

        val product_index = categories.columns.indexOf("product")
        val class_index = categories.columns.indexOf("class")
        val subclass_index = categories.columns.indexOf("subclass")

        /*
            To build the product vectors, we need to know:
                1) Subclass of each product
                2) Class of each product
                3) Class of each subclass
                4) Associated classes
                5) Associated subclasses
            The 'subclasses' array contains the distinct subclasses. Their order is also the order of the vector
            that must be produced, aligned to the customer vectors. We must augment this array to also contain the class
            of each subclass.
        */

        // We suppose that the subclass -> class array is small enough to fit into main memory
        val categories_with_rules =
            categories
                .select("subclass", "class") // Get sublcass and class
                .distinct()
                .join(subclass_rules, $"subclass" === subclass_rules.col("lhs"), "left_outer")
                .drop("lhs")
                .withColumnRenamed("rhs", "subclass_associations")
                .join(class_rules, $"class" === class_rules.col("lhs"), "left_outer")
                .drop("lhs")
                .withColumnRenamed("rhs", "class_associations")

        val subclass_class_arr = categories_with_rules.rdd // Cast to RDD
                // Map to (subclass, class, subclass associations, class associations)
                .map(x => (x.getString(0), x.getString(1), x.getAs[mutable.WrappedArray[String]](2), x.getAs[mutable.WrappedArray[String]](3)))
                .filter(x => subclasses.contains(x._1)) // Filter subclasses not in baskets
                .sortBy(x => subclasses.indexOf(x._1)) // Order by subclasses array to align with customer vectors
                .collect() // Collect since we want to use it in another rdd operation

        // Convert into RDD - tuple format (product, subclass, class)
        val categories_rdd = categories
            .rdd
            .map(x => (x.getString(product_index), x.getString(subclass_index), x.getString(class_index)))

        // Compute product vectors and cast to (product, product_vector) tuples
        categories_rdd.map(x => (x._1, {
            val product = x._1
            val product_subclass = x._2
            val product_class = x._3

            val vec = subclass_class_arr.map(x =>{
                if(x._1 == product_subclass) 1.0 // Product subclass == current subclass
                else if(x._2 == product_class) 0.5 // Product class == current class
                else if(x._3 != null && x._3.contains(product_subclass)) 1.0 // Product subclass in current subclass associations
                else if(x._4 != null && x._4.contains(product_class)) 0.25 // Product class in current class associations
                else 0.0
            })

            vec
        }))
    }

    def vectorize_customers(groceries: DataFrame, subclassCountVectorizer: CountVectorizerModel, outputCol: String, spark: SparkSession): RDD[(Int, Array[Double])] = {
        import spark.sqlContext.implicits._

        /*
            We must vectorize each product subclass basket, and add them all together to get the total spending
            of a customer per subclass. We will use a CountVectorizer, fitted in the subclasses.
         */
        val groceries_subclass_vectors = subclassCountVectorizer
            .setInputCol("subclass_groceries")
            .setOutputCol("subclass_vector")
            .transform(groceries)

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
