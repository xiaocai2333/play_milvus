# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import argparse
import time
import os
import numpy as np
import signal
import sys
import time

from threading import Timer

from pymilvus import (
    Collection, Partition,
    connections, utility
)
from common import *


sift_dir_path = "/czsdata/sift1b/"
sift_dir_path = "/test/milvus/raw_data/sift1b/"
deep_dir_path = "/czsdata/deep1b/"
deep_dir_path = "/test/milvus/raw_data/deep1b/"

IndexDone = False
globalIndexT = None
CurIndexType = ""

Spinner = spinning_cursor()

IndexTotalRows = 0
IndexRows = 0

def print_index_progress():
    global IndexTotalRows, IndexRows
    sys.stdout.write("\r")
    outputStr = "Indexing %s:[%-8d/%-9d]" % (CurIndexType, IndexRows, IndexTotalRows)
    sys.stdout.write(outputStr)
    sys.stdout.flush()

def index_time_printer():
    if not IndexDone:
        print_index_progress()
        loop_index_monitor(0.5)

def loop_index_monitor(t):
    global globalIndexT
    globalIndexT = Timer(t, index_time_printer)
    globalIndexT.start()


def close():
    global globalIndexT
    if globalIndexT:
        globalIndexT.cancel()
    print("\nend")


def connect_server(host):
    # configure milvus hostname and port
    print("connecting ...")
    connections.connect(host=host, port=19530)
    print("connected")

def confirm_collection_index(collection):
    global IndexTotalRows, IndexRows, IndexDone
    IndexDone = False
    info = utility.index_building_progress(collection.name)
    IndexRows, IndexTotalRows =  info['indexed_rows'], info['total_rows']
    #print("IndexTotalRows", IndexTotalRows)
    while IndexRows != IndexTotalRows:
        time.sleep(2)
        info = utility.index_building_progress(collection.name)
        IndexRows, IndexTotalRows =  info['indexed_rows'], info['total_rows']
        #print("IndexTotalRows", IndexTotalRows)
    print_index_progress()
    IndexDone = True
    
def create_index(collection, dataset, indextype, sync):
    global IndexDone, CurIndexType
    IndexDone = False
    modeStr = "synchronally" if sync else "asyncially"
    print("start building index %s"% modeStr)
    loop_index_monitor(0.2)
    if dataset == DATASET_DEEP:
        if indextype == IndexTypeIVF_FLAT:
            CurIndexType = IndexTypeIVF_FLAT
            create_deep_ivfflat_index(collection,sync)
        elif indextype == IndexTypeHNSW:
            CurIndexType = IndexTypeHNSW
            create_deep_hnsw_index(collection, sync)
        else:
            raise_exception("wrong indextype")
    elif dataset == DATASET_SIFT:
        if indextype == IndexTypeIVF_FLAT:
            CurIndexType = IndexTypeIVF_FLAT
            create_sift_ivfflat_index(collection, sync)
        elif indextype == IndexTypeHNSW:
            CurIndexType = IndexTypeHNSW
            create_sift_hnsw_index(collection, sync)
        else:
            raise_exception("wrong indextype")
    elif dataset == DATASET_TAIP:
        if indextype == IndexTypeIVF_FLAT:
            CurIndexType = IndexTypeIVF_FLAT
            create_taip_ivfflat_index(collection, sync)
        elif indextype == IndexTypeHNSW:
            CurIndexType = IndexTypeHNSW
            create_taip_hnsw_index(collection, sync)
        else:
            raise_exception("wrong indextype")
    else:
        raise_exception("wrong dataset")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler(close))
    parser = argparse.ArgumentParser(
        description=" Check index building..")
    parser.add_argument("--host", type=str, nargs=1,
                        help="host:xx.xx.xx.xx", required=True)

    parser.add_argument('--dataset', type=str, nargs=1,
                        help="dataset: sift | deep", required=True)

    parser.add_argument('--index', type=str, nargs='?', 
                        help="index: HNSW | IVF_FLAT", const="", default="")

    args = parser.parse_args()
    host = args.host[0]
    dataset = args.dataset[0]

    print("Host:", host)
    print("Dataset:", dataset)
    index = args.index
    if index:
        print("Index:", args.index)

    connect_server(host)
    try:
        collection = prepare_collection(dataset)
        if index == "NONE":
            collection.drop_index()
        elif index:
            start = time.time()
            print("start create index: ", start)
            create_index(collection, dataset, index, False)
            confirm_collection_index(collection)
            end = time.time()
            print("create index end: ", end)
            print("create index cost: ", end-start)
    finally:
        close()
