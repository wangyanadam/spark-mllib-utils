import sys
import time as tm
import pickle as pk
import random as rnd

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark import StorageLevel

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD

def __main__():
    input = ""
    output1 = ""
    output2 = ""
    nparts = 0
    weights = [1.0,1.0]
    batch_size = 10
    info = False

    for arg in sys.argv:
        arg_name_val = arg.split('=')

        if arg_name_val[0] == '-input':
            input = str(arg_name_val[1])
        elif arg_name_val[0] == '-nparts':
            nparts = int(arg_name_val[1])
        elif arg_name_val[0] == '-output1':
            output1 = str(arg_name_val[1])
        elif arg_name_val[0] == '-output2':
            output2 = str(arg_name_val[1])
        elif arg_name_val[0] == '-weights':
            arg_vals = arg_name_val[1].split(':')
            weights[0] = int(arg_vals[0])
            weights[1] = int(arg_vals[1])
        elif arg_name_val[0] == "-batch_size":
            batch_size = int(arg_name_val[1])
        elif arg_name_val[0] == "-info":
            if arg_name_val[1] == 'on':
                info = True

    conf = SparkConf()
    sc = SparkContext(conf=conf)

    data = sc.pickleFile(input)
    if nparts > 0:
        data = data.repartition(nparts)

    weights = sc.broadcast(weights)

    def map_split(p_iter):
        for i, p in enumerate(p_iter):
            if i % sum(weights.value) < weights.value[0]:
                yield (1, p)
            else:
                yield (2, p)

    data = data.mapPartitions(map_split) \
                .persist(StorageLevel.MEMORY_AND_DISK)

    data1 = data.filter(lambda (k, p): k == 1)
    data1.values() \
        .saveAsPickleFile(output1, batch_size)
    
    data2 = data.filter(lambda (k, p): k == 2)
    data2.values() \
        .saveAsPickleFile(output2, batch_size)

    if info:
        print('Data 1 Size = ' + str(data1.count()))
        print('Data 2 Size = ' + str(data2.count()))

if __name__ == '__main__':
    __main__()
