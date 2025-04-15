# Spark DataNodes x Optimizations experiment

Experiment on performance of a ML model training & evaluating pipeline in hadoop + spark setup with different configurations: 1 datanode vs 3 datanodes, default spark configurations vs optimized.
As an example task the anime recommendation (based on [dbdmobile/myanimelist-dataset](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)) via colaborative filtering using ALS from spark mllib.

## How to reproduce

> Note: this experiment requires docker compose to be installed

Clone the repository and go to respective subdir
```bash
git clone https://github.com/Ailurus1/spark-experiments
cd spark-experiments/datanode_opt
```

Install python dependencies (I used [uv](https://docs.astral.sh/uv/getting-started/installation/)):
```bash
uv venv --python 3.12 --python-preference only-managed datanode_opt
source datanode_opt/bin/activate
uv pip install -r pyproject.toml
```

Core script is `run.sh`. Give it permissions to be executed and then just run it:
```bash
chmod +x run.sh
./run.sh
```

It will perform two experiments: 1 datanode - default spark vs optimized; 3 datanodes - default spark vs optimized. Dataset will be downloaded automatically. Resulting tables and charts will be stored in `results_1n` and `results_3dn` directories respectively.


## Note
Docker compose files references - [docker-hadoop](https://github.com/big-data-europe/docker-hadoop), [docker-spark](https://github.com/big-data-europe/docker-spark)
