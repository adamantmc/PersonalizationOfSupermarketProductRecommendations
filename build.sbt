name := "Personalization of supermarket product recommendations"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % "2.3.0",
    "org.apache.spark" %% "spark-sql" % "2.3.0",
    "org.apache.spark" %% "spark-mllib" % "2.3.0",
    "org.apache.logging.log4j" % "log4j-core" % "2.10.0"
)

