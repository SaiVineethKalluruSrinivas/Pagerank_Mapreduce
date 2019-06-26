import re
import sys
from operator import add
from pyspark.sql import SparkSession


def getContrib(xs, pr):
    len_xs = len(xs)
    for x in xs:
        yield (x, pr/len_xs) #maintain state after creating each key

def getNeighbours(line):
    nodes = re.split(r'\s+',line)
    return nodes[0] , nodes[1]


if __name__ == "__main__":
    file_name = "/remote/us01home57/kallu/personal_projects/pagerank_pyspark/pagerank_dataset.txt"
    iterations = int(sys.argv[1])
    discount = float(sys.argv[2])
    spark = SparkSession.builder.appName("PagerankPyspark").getOrCreate()

    lines = spark.read.text(file_name).rdd.map(lambda r: r[0])
    links = lines.map(lambda urls: getNeighbours(urls)).distinct().groupByKey()
    ranks = links.map(lambda url: (url[0],1.0))

    for iteration in range(iterations):
        contributions = links.join(ranks).flatMap(lambda ys_xs_pr: getContrib(ys_xs_pr[1][0], ys_xs_pr[1][1]))
        ranks = contributions.reduceByKey(add).mapValues(lambda pr: pr*discount + (1.0-discount))

    sort_nodes = ranks.sortBy(lambda node_rank: -node_rank[1])
    for (node, rank) in sort_nodes.collect():
        print("{} has rank {}".format(node, rank))

    spark.stop() #stop session
