from pyspark.sql.functions import lit, udf, col
from pyspark.sql.types import FloatType
from pyspark.sql.dataframe import DataFrame
from pyspark.mllib.linalg.distributed import CoordinateMatrix
import math
import numpy as np


jaccard_distance = spark.read.csv("/home/your_path/ratings.csv", header=True, inferSchema=True)
jaccard_distance = jaccard_distance.select(["movieId", "userId"]).withColumn("rating", lit(1.0))

cosine_similarity_ratings = spark.read.csv("/home/your_path/ratings.csv", header=True, inferSchema=True)
cosine_similarity_ratings = cosine_similarity_ratings.select(["movieId", "userId", "rating"])


cmat_3 = CoordinateMatrix(cosine_similarity_ratings.rdd.map(tuple))

i_3 = cmat_3.toIndexedRowMatrix()
i_df_3 = i_3.rows.toDF(["id", "features"])

def transform(self, f):
    return f(self)

DataFrame.transform = transform

closeEnough = udf(jaccardSimilarity, FloatType())

distance_cosine = udf(cosineSimilarity, FloatType())

# possibleMatches = i_df_3.transform(lambda df: df.alias("left").crossJoin(df.alias("right")))
possibleMatches = i_df_3.transform(lambda df: df.alias("left").join(df.alias("right")))

result = possibleMatches.filter((col("left.id") != col("right.id")) & (col("left.id") < col("right.id")) ) \
        .withColumn("new", closeEnough("left.features", "right.features")) \
        .select("left.id", "right.id", "new")

result = possibleMatches.filter((col("left.id") != col("right.id")) & (col("left.id") < col("right.id")) ) \
        .withColumn("cosine", distance_cosine("left.features", "right.features")) \
        .select("left.id", "right.id", "cosine")
    

def jaccardSimilarity(v1, v2):
    indices1 = set(v1.indices)
    indices2 = set(v2.indices)
    intersection = set.intersection(indices1, indices2)
    union = indices1.union(indices2)
    return (float(len(intersection)) / float(len(union)))


def cosineSimilarity(v1, v2):
    values1 = v1.values
    values2 = v2.values
    dotProduct = sum(sum(np.outer(v1.values, v2.values)))
    ratingNorm = math.sqrt(sum(values1 ** 2))
    rating2Norm = math.sqrt(sum(values2 ** 2))
    return (float(dotProduct / (ratingNorm * rating2Norm)))
