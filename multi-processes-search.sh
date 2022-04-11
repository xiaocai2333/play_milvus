#!/bin/bash


for i in {1..10}
do
	echo $i
	nohup python milvus_search.py --host localhost --dataset taip --index HNSW  --process 1 > 5M_parallel_$i".txt" &
done
