
def load_df(file: String, schema: org.apache.spark.sql.types.StructType = null): org.apache.spark.sql.DataFrame = {
  val sql = new org.apache.spark.sql.SQLContext(sc)
  val options = Map(
    "path" -> file, 
    "header" -> "true",
    "inferSchema" -> "true")  
  /**
  schema match {
    case null => return sql.load("com.databricks.spark.csv", options)
    case _ => return sql.load("com.databricks.spark.csv", schema, options)    
  }
  */
  return sql.load("com.databricks.spark.csv", schema, options)
}