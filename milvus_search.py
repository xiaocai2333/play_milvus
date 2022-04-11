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
import threading

from threading import Timer
import numpy as np
from pymilvus import (
    Collection, Partition,
    connections, utility
)
from pymilvus.orm.types import CONSISTENCY_EVENTUALLY

from common import *

sift_dir_path = "/czsdata/sift1b/"
sift_dir_path = "/test/milvus/raw_data/sift1b/"
deep_dir_path = "/czsdata/deep1b/"
deep_dir_path = "/test/milvus/raw_data/deep1b/"
taip_dir_path = "/data/milvus/raw_data/zjlab"

# EF_SEARCHS = [50, 64, 80, 100, 128, 168, 200, 256]
# EF_SEARCHS = [100, 128, 168, 200, 256]
EF_SEARCHS = [50]
# NPROBES = [4, 6, 8, 12, 16, 20, 24, 32, 40, 50, 64, 128]
NPROBES = [1, 4, 6, 8, 16]

TOPK = 50
BNQ = 10000
NQ = [1]
# NQ = [1,10,100,1000,10000]
QueryFName = "query.npy"
RUN_NUM = 10000
ALL_QPS = 0.0

Spinner = spinning_cursor()

def connect_server(host):
    connections.connect(host=host, port=19530)
    print(f"connected")


def get_recall(r1, r2):
    recall = 0.0
    for i in range(len(r2)):
        count = np.intersect1d(r1[i], r2[i]).shape[0]
        recall += count/len(r2[i])
    recall = recall/len(r2)
    return recall


# def search_collection(host, dataset, indextype):
def search_collection(queryData, indextype):
    global ALL_QPS
    # connect_server(host)
    # collection = prepare_collection(dataset)
    query_fname = ""
    metric_type = ""
    # if dataset == DATASET_DEEP:
    #     metric_type = "IP"
    #     query_fname = os.path.join(deep_dir_path, QueryFName)
    # elif dataset == DATASET_SIFT:
    #     query_fname = os.path.join(sift_dir_path, QueryFName)
    #     metric_type = "L2"
    # elif dataset == DATASET_TAIP:
    #     query_fname = os.path.join(taip_dir_path, QueryFName)
    #     metric_type = "L2"
    #
    # if metric_type == "" or query_fname == "":
    #     raise_exception("wrong dataset")
    #
    # queryData = np.load(query_fname)
    # print(queryData)
    metric_type = "L2"
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
        query_list = queryData.tolist()[:BNQ]
        run_counter = 0
        run_time = 0
        result = []
        while (run_counter < 1):
            start = time.time()
            result = collection.search(query_list, "vec", search_params, TOPK, consistency_level=CONSISTENCY_EVENTUALLY)
            search_time = time.time() - start
            run_time = run_time + search_time
            run_counter = run_counter + 1
            print("search time cost:", search_time)
        aver_time = run_time * 1.0
        qps = BNQ * 1.0 / aver_time
        fmt_str = "average_time, qps: "
        print(fmt_str)
        print(aver_time, qps)
        all_ids = []
        for hits in result:
            all_ids.append(list(hits.ids))
            # print(hits.distances)
        with open("result.txt", 'w') as f:
            f.write(json.dumps(all_ids))
        return

    if not plist:
        raise_exception("wrong dataset")

    # f = open("./result.txt")
    # dataset_result = f.read()
    # f.close()
    # dataset_result = json.loads(dataset_result)
    for nq in NQ:
        query_list = queryData.tolist()[:nq]
        for s_p in plist:
            run_counter = 0
            run_time = 0
            result = []
            while(run_counter < RUN_NUM):
                start = time.time()
                search_params["params"][param_key] = s_p
                result = collection.search(query_list, "vec", search_params, TOPK, consistency_level=CONSISTENCY_EVENTUALLY)
                search_time = time.time() - start
                run_time = run_time + search_time
                run_counter = run_counter + 1
                # print("search cost:", search_time)
                # for re in result:
                #     print(re.ids)
                #     print(re.distances)
                # all_ids = []
                # for hits in result:
                #     all_ids.append(list(hits.ids))
                    # print(hits.distances)
                # print(all_ids)
            aver_time = run_time * 1.0 / RUN_NUM
            qps = nq*1.0/aver_time
            ALL_QPS = ALL_QPS + qps
            print("nq: %s, %s: %s" % (nq, param_key, s_p))
            # print("average_time\t, vps\t, recall: ")
            print("average_time\t, vps\t")
            # print(aver_time, qps, get_recall(dataset_result[:nq], all_ids))
            print(aver_time, qps)

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
    parser.add_argument('--process', type=int, nargs=1,
                        help="process: 1...", required=True)

    args = parser.parse_args()
    host = args.host[0]
    dataset = args.dataset[0]
    indextype = args.index[0]
    process_num = args.process[0]

    print("Host:", host)
    print("Dataset:", dataset)
    print("IndexType", indextype)
    print("ProcessNum", process_num)

    connect_server(host)
    collection = prepare_collection(dataset)

    query_fname = os.path.join(taip_dir_path, QueryFName)
    queryData = np.load(query_fname)

    search_processes = []
    # for i in range(process_num):
    #     search_processes.append(multiprocessing.Process(target=search_collection,
    #                                                     args=(host, dataset, indextype)))
    for i in range(process_num):
        search_processes.append(threading.Thread(target=search_collection,
                                                 args=(queryData, indextype)))
    for p in search_processes:
        p.start()
    for p in search_processes:
        p.join()
    # search_collection(collection, dataset, indextype)
