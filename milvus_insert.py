# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import argparse
import time
import os

import signal
import sys

from threading import Timer
import numpy as np

from pymilvus import (
    list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection, Partition,
    connections,
)
from common import *

ID_COUNTER = 0
NUM_FILES = 2
PARTITION_NUM = 1

sift_dir_path = "/czsdata/sift1b/"
sift_dir_path = "/test/milvus/raw_data/sift1b/"
deep_dir_path = "/czsdata/deep1b/"
deep_dir_path = "/test/milvus/raw_data/deep1b/"
taip_dir_path = "/data/milvus/raw_data/zjlab"

InsertDone = False

# 100k per file
PER_FILE_ROWS = 100000

globalInsertT = None

Spinner = spinning_cursor()

CurPartitionName = DEFAULT_PARTITION_NAME

PartitionTotal = 20
PartitionCur = 0

Total = 100
Cur = 0

def print_insert_progress():
    sys.stdout.write("\r")
    sys.stdout.write("Partition:%-9s Inserting:[%8d/%9d] Total:[%-9d/%-9d]" % (
        CurPartitionName, PartitionCur, PartitionTotal, Cur, Total))
    sys.stdout.flush()

def insert_time_printer():
    if not InsertDone:
        if insert_time_printer.cur_partition != CurPartitionName:
            if insert_time_printer.cur_partition != "":
                sys.stdout.write("\r\n")
            insert_time_printer.cur_partition = CurPartitionName
        print_insert_progress()
        loop_monitor(1)


insert_time_printer.cur_partition = DEFAULT_PARTITION_NAME

def loop_monitor(t):
    global globalInsertT
    globalInsertT = Timer(t, insert_time_printer)
    globalInsertT.start()


def gen_fnames(fmt, start, end):
    ret = []
    for i in range(start, end):
        ret.append(fmt % i)
    return ret


def gen_deep1b_fnames(start, end):
    fmt = os.path.join(deep_dir_path, "binary_96d_%05d.npy")
    return gen_fnames(fmt, start, end)


def gen_taip1b_fnames(start, end):
    fmt = os.path.join(taip_dir_path, "binary_768d_%05d.npy")
    return gen_fnames(fmt, start, end)


def gen_sift1b_fnames(start, end):
    fmt = os.path.join(sift_dir_path, "binary_128d_%05d.npy")
    return gen_fnames(fmt, start, end)


def close():
    global globalInsertT
    if globalInsertT:
        globalInsertT.cancel()
    print("\nend")


def insert_dataset(collection, num, partition_num, gen_fnames_f):
    if not callable(gen_fnames_f):
        raise_exception("pass wrong function in insert_dataset")

    global PartitionCur, PartitionTotal, Cur, Total, CurPartitionName
    if num % partition_num != 0:
        raise_exception("num %% partition_num must be zero")

    partition_names = ["p%d" % i for i in range(partition_num)]
    partition_names[0] = DEFAULT_PARTITION_NAME
    cnt = num // partition_num
    PartitionTotal = cnt * PER_FILE_ROWS * 4
    Total = PER_FILE_ROWS * num * 4
    for i, p_name in enumerate(partition_names, 0):
        CurPartitionName = p_name
        PartitionCur = 0
        start = i * cnt
        end = start + cnt
        fnames = gen_fnames_f(start, end)
        if p_name != DEFAULT_PARTITION_NAME:
            partition = collection.create_partition(p_name)
        for _ in range(4):
            for fname in fnames:
                insert_afile_to_collection(collection, fname, p_name)
                PartitionCur += PER_FILE_ROWS
                Cur += PER_FILE_ROWS
        time.sleep(1)


def insert_afile_to_collection(collection, fname, partition_name):
    global ID_COUNTER
    data = np.load(fname)
    block_size = PER_FILE_ROWS
    entities = [
        [i for i in range(ID_COUNTER, ID_COUNTER + block_size)],
        data.tolist()
    ]

    insert_result = collection.insert(entities, partition_name=partition_name)
    if insert_result.insert_count != block_size:
        raise_exception("insert failed:%s" % fname)

    ID_COUNTER = ID_COUNTER + block_size


def insert_sift_dataset(collection, num, partition_num):
    insert_dataset(collection, num, partition_num, gen_sift1b_fnames)


def insert_deep_dataset(collection, num, partition_num):
    insert_dataset(collection, num, partition_num, gen_deep1b_fnames)


def insert_taip_dataset(collection, num, partition_num):
    insert_dataset(collection, num, partition_num, gen_taip1b_fnames)


def connect_server(host):
    # configure milvus hostname and port
    print(f"\nCreate connection...")
    connections.connect(host=host, port=19530)

def prepare_collection(dataset):
    collection_name = dataset
    if dataset == DATASET_DEEP:
        dim = 96
    elif dataset == DATASET_SIFT:
        dim = 128
    elif dataset == DATASET_TAIP:
        dim = 768
    else:
        raise_exception("wrong dataset")

    # List all collection names
    print(f"\nList collections...")
    collection_list = list_collections()
    print(list_collections())

    if (collection_list.count(collection_name)):
        print(collection_name, " exist, and drop it")
        collection = Collection(collection_name)
        collection.drop()
        print("drop collection ", collection_name)

    field1 = FieldSchema(name="id", dtype=DataType.INT64, description="int64", is_primary=True)
    field2 = FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, description="float vector", dim=dim, is_primary=False)
    schema = CollectionSchema(fields=[field1, field2], description="")
    collection = Collection(name=collection_name, data=None, schema=schema, shards_num=2)
    return collection


def confirm_collection_insert(collection):
    nums = collection.num_entities
    global InsertDone
    InsertDone = True

    if globalInsertT:
        globalInsertT.cancel()

    print("\nnumber_entities:", nums)

def insert_collection(collection, dataset):
    loop_monitor(1)
    if dataset == DATASET_DEEP:
        insert_deep_dataset(collection, NUM_FILES, PARTITION_NUM)
    elif dataset == DATASET_SIFT:
        insert_sift_dataset(collection, NUM_FILES, PARTITION_NUM)
    elif dataset == DATASET_TAIP:
        insert_taip_dataset(collection, NUM_FILES, PARTITION_NUM)
    else:
        raise_exception("wrong dataset")
    
def create_index(collection, dataset, indextype):
    if dataset == DATASET_DEEP:
        if indextype == IndexTypeIVF_FLAT:
            create_deep_ivfflat_index(collection, False)
        elif indextype == IndexTypeHNSW:
            create_deep_hnsw_index(collection, False)
        elif indextype == "NONE":
            print("do not create index")
        else:
            raise_exception("wrong indextype")
    elif dataset == DATASET_SIFT:
        if indextype == IndexTypeIVF_FLAT:
            create_sift_ivfflat_index(collection, False)
        elif indextype == IndexTypeHNSW:
            create_sift_hnsw_index(collection, False)
        elif indextype == "NONE":
            print("do not create index")
        else:
            raise_exception("wrong indextype")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler(close))
    parser = argparse.ArgumentParser(
        description="Insert Data to Milvus user-defined ranges of .npy files")

    parser.add_argument("--host", type=str, nargs=1,
                        help="host:xx.xx.xx.xx", required=True)

    parser.add_argument('--dataset', type=str, nargs=1,
                        help="dataset: sift | deep", required=True)

    parser.add_argument('--index', type=str, nargs=1, 
                        help="index: HNSW | IVF_FLAT | NONE", required=True)

    args = parser.parse_args()
    host = args.host[0]
    dataset = args.dataset[0]
    indextype = args.index[0]

    print("Host:", host)
    print("Dataset:", dataset)
    print("IndexType", indextype)

    connect_server(host)
    try:
        collection = prepare_collection(dataset)
        create_index(collection, dataset, indextype)
        insert_collection(collection, dataset)
        confirm_collection_insert(collection)
    finally:
        close()
