# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import argparse
import time
import os
import numpy as np
import signal
import sys

from threading import Timer

from pymilvus import (
    list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection, Partition,
    connections, utility
)

from common import *


sift_dir_path = "/czsdata/sift1b/"
sift_dir_path = "/test/milvus/raw_data/sift1b/"
deep_dir_path = "/czsdata/deep1b/"
deep_dir_path = "/test/milvus/raw_data/deep1b/"

LoadDone = False
globalLoadT = None

Spinner = spinning_cursor()

def print_load_progress():
    sys.stdout.write("\r")
    sys.stdout.write("Loading [%s]" % next(Spinner))
    sys.stdout.flush()

def load_time_printer():
    if not LoadDone:
        print_load_progress()
        loop_load_monitor(0.5)

def loop_load_monitor(t):
    global globalLoadT
    globalLoadT = Timer(t, load_time_printer)
    globalLoadT.start()


def close():
    global globalLoadT
    if globalLoadT:
        globalLoadT.cancel()
    print("\nend")


def connect_server(host):
    # configure milvus hostname and port
    print("connecting ...")
    connections.connect(host=host, port=19530)
    print("connected")

def parse_collection_info(infos):
    mapInfo = {}
    for info in infos:
        nodeInfo = mapInfo.setdefault(info.nodeID, {})

        indexed = info.index_name != ""
        stateStr = "Invalid"
        if indexed:
            stateStr = "Indexed"
        else:
            if info.state == 2:
                stateStr = "Growing"
            elif info.state == 3:
                stateStr = "Sealed"

        stateInfo = nodeInfo.setdefault(stateStr, {})
        partitionID = info.partitionID
        partitionNumRows = stateInfo.setdefault(partitionID, 0) 
        partitionNumRows += info.num_rows
        stateInfo[partitionID] = partitionNumRows
   
    for nId, nInfo in mapInfo.items():
        n_total = 0
        for st, stInfo in nInfo.items():
            st_total = sum(stInfo.values())
            stInfo["total"] = st_total
            n_total += st_total
        nInfo["total"] = n_total

    return mapInfo

def print_map_info(mapInfo):
    total = 0
    for nId, nInfo in mapInfo.items():
        if nInfo:
            print("NodeID:", nId)
        for st, stInfo in nInfo.items():
            if st == "total":
                print("\t %s :%d"%(st,stInfo))
            else:
                print("\t %s"%st)
                for p, pnum in stInfo.items():
                    print("\t\t%s:%d"%(p,pnum))
        total += nInfo["total"]

    print("total:", total)

def confirm_collection_load(collection):
    if globalLoadT:
        globalLoadT.cancel()
    collection_info = utility.get_query_segment_info(collection.name)
    mapInfo = parse_collection_info(collection_info)
    print_map_info(mapInfo)
    
def load_collection(collection, dataset, partitions):
    print("start to load")
    global LoadDone, globalLoadT
    if dataset not in (DATASET_DEEP, DATASET_SIFT):
        raise_exception("wrong dataset")
    loop_load_monitor(1)
    collection = Collection(name=dataset)
    collection.release()
    collection.load(partition_names=partitions)
    LoadDone = True
    if globalLoadT:
        globalLoadT.cancel()
    print("load done")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler(close))
    parser = argparse.ArgumentParser(
        description="Insert Data to Milvus user-defined ranges of .npy files")
    parser.add_argument("--host", type=str, nargs=1,
                        help="host:xx.xx.xx.xx", required=True)

    parser.add_argument('--dataset', type=str, nargs=1,
                        help="dataset: sift | deep", required=True)

    parser.add_argument('--reload', action='store_true')

    parser.add_argument('-p', action='append',
                        help="p: p0 p:p1", required=False, default=[])

    args = parser.parse_args()
    host = args.host[0]
    dataset = args.dataset[0]
    need_reload = args.reload
    partitions = args.p or None

    print("Host:", host)
    print("Dataset:", dataset)
    if need_reload and partitions:
        print("Partitions:", partitions)
    connect_server(host)
    try:
        collection = prepare_collection(dataset)
        if need_reload:
            load_collection(collection, dataset, partitions)
        confirm_collection_load(collection)
    finally:
        close()
