# BERT-fine-tuning-analysis
The codebase for the paper: A Closer Look at How Fine-tuning Changes BERT.

# Installing
This codebase is dervied from the [DirectProbe][], following
the same install instructions as [DirectProbeCode][].

# Getting Started

## Download datasets and Running examples

1. Download the pre-packed data from [here][data_url] and
   unzip them. The data format is the same as [DirectProbeCode][].
2. Suppose all the pre-packed data is put in the directory
   `data`, then we can run an experiment using the
   configuration from `config.ini`.

    ```
        python main.py
    ```

## Results
After probing, you will find the results in the
directory `results/SS/`.(We are using the supersense
role task as the example.)
In this directory, there are 6 files:
- `clusters.txt`: The clustering results. Each line contains
  a cluster number for the corresponding training example. 

- `inside_max.txt`: The maximum pairwise distances inside
  each cluster. Each line represents one cluster.

- `inside_mean.txt`: The mean pairwise distances inside each
  cluster. Each line represents one cluster.

- `log.txt`: The probing log file.

- `outside_min.txt`: The minimum distance to other clusters
  for each cluster. Each line represents one cluster.

- `vec.txt`: Pairwise distances between clusters. Each line
  represents a pair of cluster and its distance.

# Citations



[DirectProbe]: https://aclanthology.org/2021.naacl-main.401/
[DirectProbeCode]: https://github.com/utahnlp/DirectProbe
[data_url]: https://drive.google.com/drive/folders/1mlF-O20Zsa_jJG3tjV-vVrIivY71_R5P
