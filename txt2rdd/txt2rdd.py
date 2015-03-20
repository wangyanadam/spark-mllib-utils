import re
import sys
import numpy as np
import random as rnd
import pickle as pk

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark import StorageLevel

from pyspark.mllib.regression import LabeledPoint

def __main__():
    
    # setting args
    input = ""
    batch_size = 10
    output_rdd = ""
    nparts = 1
    delims = ',|\t| '
    lcol = -1
    real_cols = []
    cate_cols = []
    null_value = ''
    # hashing categorical features = {none, hex8, bin32}
    hash = 'none'
    # missing value substitution = {zero,mean}
    missing_value = 'zero'

    for arg in sys.argv:
        arg_name_val = arg.split('=')
        
        if arg_name_val[0] == '-input':
            input = str(arg_name_val[1])
        elif arg_name_val[0] == '-output':
            output_rdd = str(arg_name_val[1])
        elif arg_name_val[0] == '-nparts':
            nparts = int(arg_name_val[1])
        elif arg_name_val[0] == '-delims':
            delims = str(arg_name_val[1])
        elif arg_name_val[0] == '-lcol':
            lcol = int(arg_name_val[1])
        elif arg_name_val[0] == "-batch_size":
            batch_size = int(arg_name_val[1])
        elif arg_name_val[0] == '-real_cols':
            for idx_range in arg_name_val[1].split(','):
                idx_nums = idx_range.split('-')
                if len(idx_nums) == 2:
                    real_cols.extend(range(int(idx_nums[0]), int(idx_nums[1]) + 1))
                else:
                    real_cols.append(int(idx_nums[0]))
        elif arg_name_val[0] == '-cate_cols':
            for idx_range in arg_name_val[1].split(','):
                idx_nums = idx_range.split('-')
                if len(idx_nums) == 2:
                    cate_cols.extend(range(int(idx_nums[0]), int(idx_nums[1]) + 1))
                else:
                    cate_cols.append(int(idx_nums[0]))
        elif arg_name_val[0] == '-null_value':
            null_value = str(arg_name_val[1])
        elif arg_name_val[0] == '-hash':
            hash = str(arg_name_val[1])
            if hash not in {'none', 'hex8', 'bin32'}:
                print >> sys.stderr, 'Invalid option for hash!'
                exit(-1)
        elif arg_name_val[0] == '-missing_value':
            missing_value = str(arg_name_val[1])
            if missing_value not in {'zero', 'mean'}:
                print >> sys.stderr, 'Invalid option for missing_value!'
                exit(-1)

    real_cols = set(real_cols)
    cate_cols = set(cate_cols)
    ncols = len(real_cols) + len(cate_cols)

    if (lcol >= 0) & (lcol not in real_cols):
        print >> sys.stderr, 'Invalid value type for response variable!'
        exit(-1)

    if len(real_cols.intersection(cate_cols)) != 0:
        print >> sys.stderr, 'Each feature column should be either real or categorical, but not both!'
        exit(-1)

    conf = SparkConf()
    sc = SparkContext(conf=conf)

    real_cols = sc.broadcast(real_cols)
    cate_cols = sc.broadcast(cate_cols)

    # mapping each text line to separated fields
    def map_txt2val(line):
        fields = re.split(delims, line)
        mapped_fields = [None] * len(fields)

        for real_col in real_cols.value:
            if fields[real_col] != '':
                mapped_fields[real_col] = float(fields[real_col])
            else:
                mapped_fields[real_col] = None

        for cate_col in cate_cols.value:
            if fields[cate_col] != '':
                mapped_fields[cate_col] = str(fields[cate_col])
            else:
                mapped_fields[cate_col] = None

        return mapped_fields

    data0 = sc.textFile(input, nparts) \
            .filter(lambda line: len(re.split(delims, line)) == ncols) \
            .map(map_txt2val) \
            .persist(StorageLevel.MEMORY_ONLY)

    cate_feat_stats = {key:None for key in cate_cols.value}

    if hash == 'none':
        for i in cate_cols.value:
            cate_feat_stats[i] = data0.map(lambda fields: fields[i]) \
                                    .filter(lambda field: field != None) \
                                    .distinct() \
                                    .collect()

    cate_feat_stats = sc.broadcast(cate_feat_stats)

    # mapping categorical value to reals
    def map_cate2real(fields):
        mapped_fields = []
        
        for i,f in enumerate(fields):
            if i in cate_cols.value:
                encode = None
                
                if f == None:
                    if hash == 'none':
                        encode = [None] * len(cate_feat_stats.value[i])
                    elif hash == 'hex8':
                        encode = [None] * 32
                    elif hash == 'bin32':
                        encode = [None] * 32
                else:
                    if hash == 'none':
                        encode = [0] * len(cate_feat_stats.value[i])
                        encode[cate_feat_stats.value[i].index(f)] = 1
                    elif hash == 'hex8':
                        encode = [int(b) for b in (list(bin(int(f, 16)))[2:])]
                        encode = [0] * (32 - len(encode)) + encode
                    elif hash == 'bin32':
                        encode = [int(b) for b in list(f)]
                        encode = [0] * (32 - len(encode)) + encode
                    
                mapped_fields += encode
            else:
                mapped_fields += [f]
        
        return mapped_fields

    data1 = data0.map(map_cate2real) \
                .persist(StorageLevel.MEMORY_ONLY)
    data0.unpersist()
    
    # Update the index of the label column
    if lcol >= 0:
        shift = 0
        for i in range(0, lcol):
            if i in cate_cols.value:
                if hash == 'none':
                    shift += len(cate_feat_stats.value[i])
                elif hash == 'hex8':
                    shift += 32
                elif hash == 'bin32':
                    shift += 32
            else:
                shift += 1
        lcol = shift


    def map_sum_count(fields):
        mapped_fields = [None] * len(fields)
        for i,f in enumerate(fields):
            if f == None:
                mapped_fields[i] = (0, 0)
            else:
                mapped_fields[i] = (f, 1)
        return mapped_fields

    def reduce_sum_count(a,b):
        c = []
        for aa, bb in zip(a, b):
            c += [(aa[0] + bb[0], aa[1] + bb[1])]
        return c

    sums_and_counts = data1.map(map_sum_count) \
                        .reduce(reduce_sum_count)

    filling_value = [None] * len(sums_and_counts)
    for i in range(0, len(filling_value)):
        if missing_value == 'mean':
            if sums_and_counts[i][1] != 0:
                filling_value[i] = float(sums_and_counts[i][0]) / sums_and_counts[i][1]
            else:
                filling_value[i] = 0
        elif missing_value == 'zero':
            filling_value[i] = 0
    
    filling_value = sc.broadcast(filling_value)

    def map_missing(fields):
        mapped_fields = [None] * len(fields)
        for i,f in enumerate(fields):
            if f == None:
                mapped_fields[i] = filling_value.value[i]
            else:
                mapped_fields[i] = f
        if lcol >= 0:
            return LabeledPoint(mapped_fields[lcol], mapped_fields[:lcol] + mapped_fields[lcol + 1:])
        else:
            return LabeledPoint(-1, mapped_fields)

    data1.map(map_missing) \
        .saveAsPickleFile(output_rdd, batch_size)
            
if __name__ == '__main__':
    __main__()
