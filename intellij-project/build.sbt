fork in run := true

lazy val root = (project in file("."))
  .settings(
    name := "aclimdb",

    version := "0.1.0-SNAPSHOT",

    scalaVersion := "2.11.8",

    sparkVersion := "2.2.0",

    unmanagedClasspath in Compile ++= (file("/usr/local/lib/spark/jars") ** "*.jar").classpath,

    unmanagedClasspath in Runtime ++= (file("/usr/local/lib/spark/jars") ** "*.jar").classpath
  )
val sparkVersion = settingKey[String]("Spark version")
