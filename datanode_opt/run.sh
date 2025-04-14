#!/bin/bash

set -e

mkdir -p data results_1dn results_3dn

echo "Downloading dataset..."
python3 download_dataset.py

wait_for_hadoop() {
    echo "Waiting for Hadoop services to be ready..."
    while ! curl -s "http://localhost:9870" > /dev/null; do
        sleep 1
    done
    echo "Hadoop is ready!"
}

wait_for_spark() {
    echo "Waiting for Spark services to be ready..."
    while ! curl -s "http://localhost:8080" > /dev/null; do
        sleep 1
    done
    echo "Spark is ready!"
}

copy_to_hdfs() {
    echo "Creating HDFS directories..."
    docker exec namenode hdfs dfs -mkdir -p /data
    docker cp data/final_animedataset.csv namenode:/tmp/
    docker exec namenode hdfs dfs -put /tmp/final_animedataset.csv /data/
}

run_experiments() {
    local result_dir=$1

    echo "Running experiments..."
    python3 app.py
    python3 app.py --opt
    python3 analyze.py

    mv perf.csv perf_opt.csv total_performance.png function_performance.png "$result_dir/"
}

cleanup_docker() {
    echo "Stopping and removing containers..."
    docker-compose -f "$1" down -v
    sleep 5
}

echo "Starting 1 DataNode experiment..."
docker-compose -f docker-compose-1dn.yml up -d

wait_for_hadoop
wait_for_spark
copy_to_hdfs
run_experiments "results_1dn"

cleanup_docker "docker-compose-1dn.yml"

echo "Starting 3 DataNode experiment..."
docker-compose -f docker-compose-3dn.yml up -d
wait_for_hadoop
wait_for_spark
copy_to_hdfs
run_experiments "results_3dn"

cleanup_docker "docker-compose-3dn.yml"

echo "Results for 1 DataNode are in results_1dn/"
echo "Results for 3 DataNodes are in results_3dn/"
