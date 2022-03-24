# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import json
import time
import os
import signal
import sys
import multiprocessing

from threading import Timer
import numpy as np
from pymilvus import (
    Collection, Partition,
    connections, utility
)

from common import *

sift_dir_path = "/czsdata/sift1b/"
sift_dir_path = "/test/milvus/raw_data/sift1b/"
deep_dir_path = "/czsdata/deep1b/"
deep_dir_path = "/test/milvus/raw_data/deep1b/"
taip_dir_path = "/data/milvus/raw_data/zjlab"

# EF_SEARCHS = [50, 64, 80, 100, 128, 168, 200, 256]
EF_SEARCHS = [150]
# NPROBES = [4, 6, 8, 12, 16, 20, 24, 32, 40, 50, 64, 128]
NPROBES = [16]

TOPK = 50
NQ = 100
QueryFName = "query.npy"
RUN_NUM = 5
PROCESS_NUM = 2

Spinner = spinning_cursor()

def connect_server(host):
    connections.connect(host=host, port=19530)
    print(f"connected")


def get_recall(r1, r2):
    recall = 0.0
    for i in range(len(r1)):
        count = np.intersect1d(r1[i], r2[i]).shape[0]
        recall += count/len(r1[i])
    recall = recall/len(r1)
    return recall


def search_collection(collection, dataset, indextype):
    query_fname = ""
    metric_type = ""
    if dataset == DATASET_DEEP:
        metric_type = "IP"
        query_fname = os.path.join(deep_dir_path, QueryFName)
    elif dataset == DATASET_SIFT:
        query_fname = os.path.join(sift_dir_path, QueryFName)
        metric_type = "L2"
    elif dataset == DATASET_TAIP:
        query_fname = os.path.join(taip_dir_path, QueryFName)
        metric_type = "L2"

    if metric_type == "" or query_fname == "":
        raise_exception("wrong dataset")

    queryData = np.load(query_fname)
    query_list = queryData.tolist()[:NQ]
    f = open("./result.txt")
    dataset_result = f.read()
    f.close()
    dataset_result = json.loads(dataset_result)

    search_params = {"metric_type": metric_type, "params": {}}
    param_key = ""
    plist = []
    if indextype == IndexTypeIVF_FLAT:
        param_key = "nprobe"
        plist = NPROBES 
    elif indextype == IndexTypeHNSW:
        param_key = "ef"
        plist = EF_SEARCHS
    elif indextype == "NONE":
        run_counter = 0
        run_time = 0
        result = []
        while (run_counter < RUN_NUM):
            start = time.time()
            result = collection.search(query_list, "vec", search_params, TOPK, guarantee_timestamp=1)
            search_time = time.time() - start
            run_time = run_time + search_time
            run_counter = run_counter + 1
            print("search cost:", search_time)
        aver_time = run_time * 1.0 / RUN_NUM
        fmt_str = "average_time, qps: "
        print(fmt_str)
        print(aver_time, NQ * 1.0 / aver_time)
        all_ids = []
        for nq in result:
            all_ids.append(list(nq.ids))
        with open("result.txt", 'w') as f:
            f.write(json.dumps(all_ids))

        print(all_ids)
        return

    if not plist:
        raise_exception("wrong dataset")

    for s_p in plist:
        run_counter = 0
        run_time = 0
        result = []
        while(run_counter < RUN_NUM):
            start = time.time()
            search_params["params"][param_key] = s_p
            result = collection.search(query_list, "vec", search_params, TOPK, guarantee_timestamp=1)
            search_time = time.time() - start
            run_time = run_time + search_time 
            run_counter = run_counter + 1
            print("search cost:", search_time)
        all_ids = []
        for nq in result:
            all_ids.append(list(nq.ids))
        aver_time = run_time * 1.0 / RUN_NUM

        fmt_str = "%s: %s, average_time, qps, recall: "%(param_key, s_p)
        print(fmt_str)
        print(aver_time, NQ*1.0/aver_time, get_recall(dataset_result[:NQ], all_ids))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler(None))
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
    collection = prepare_collection(dataset)

    search_processes = []
    for i in range(PROCESS_NUM):
        search_processes.append(multiprocessing.Process(target=search_collection,
                                                        args=[collection, dataset, indextype]))
    for p in search_processes:
        p.start()
    for p in search_processes:
        p.join()
    # search_collection(collection, dataset, indextype)
