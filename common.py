# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
from pymilvus import (
    Collection,
)

def signal_handler(*args, callback=None):
    def outerFunc(*args):
        print('\nYou pressed Ctrl+C!')
        if callable(callback):
            callback()
        sys.exit(0)

    return outerFunc

def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor

IndexTypeHNSW = 'HNSW'
IndexTypeIVF_FLAT = 'IVF_FLAT'

DATASET_SIFT = "sift"
DATASET_DEEP = "deep"
DATASET_TAIP = "taip"

DEFAULT_PARTITION_NAME = "_default"

def raise_exception(msg):
    raise (Exception(msg))

def prepare_collection(dataset):
    if dataset not in (DATASET_DEEP, DATASET_SIFT, DATASET_TAIP):
        raise_exception("wrong dataset")

    collection = Collection(name=dataset)
    return collection 

def create_sift_hnsw_index(collection, sync):
    _async = not sync
    future = collection.create_index(field_name="vec",
                            _async = _async,
                            sync = sync,
                            index_params={'index_type': IndexTypeHNSW,
                                          'metric_type': 'L2',
                                          'params': {
                                              "M": 16,  # int. 4~64
                                              "efConstruction": 250  # int. 8~512
                                          }})
    future.done()


def create_deep_hnsw_index(collection, sync):
    _async = not sync
    future = collection.create_index(field_name="vec",
                            _async = _async,
                            sync = sync,
                            index_params={'index_type': IndexTypeHNSW,
                                          'metric_type': 'IP',
                                          'params': {
                                              "M": 12,  # int. 4~64
                                              "efConstruction": 150  # int. 8~512
                                          }})
    future.done()


def create_sift_ivfflat_index(collection, sync):
    _async = not sync
    future = collection.create_index(field_name="vec",
                            _async = _async,
                            sync = sync,
                            index_params={'index_type': IndexTypeIVF_FLAT,
                                          'metric_type': 'L2',
                                          'params': {
                                              "nlist": 8192,
                                          }})
    future.done()


def create_deep_ivfflat_index(collection, sync):
    _async = not sync
    future = collection.create_index(field_name="vec",
                            _async = _async,
                            sync = sync,
                            index_params={'index_type': IndexTypeIVF_FLAT,
                                          'metric_type': 'IP',
                                          'params': {
                                              "nlist": 8192,  # int. 4~64
                                          }})
    future.done()


