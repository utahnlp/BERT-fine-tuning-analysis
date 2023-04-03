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

```
@inproceedings{zhou-srikumar-2022-closer,
    title = "A Closer Look at How Fine-tuning Changes {BERT}",
    author = "Zhou, Yichu  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.75",
    doi = "10.18653/v1/2022.acl-long.75",
    pages = "1046--1061",
    abstract = "Given the prevalence of pre-trained contextualized representations in today{'}s NLP, there have been many efforts to understand what information they contain, and why they seem to be universally successful. The most common approach to use these representations involves fine-tuning them for an end task. Yet, how fine-tuning changes the underlying embedding space is less studied. In this work, we study the English BERT family and use two probing techniques to analyze how fine-tuning changes the space. We hypothesize that fine-tuning affects classification performance by increasing the distances between examples associated with different labels. We confirm this hypothesis with carefully designed experiments on five different NLP tasks. Via these experiments, we also discover an exception to the prevailing wisdom that {``}fine-tuning always improves performance{''}. Finally, by comparing the representations before and after fine-tuning, we discover that fine-tuning does not introduce arbitrary changes to representations; instead, it adjusts the representations to downstream tasks while largely preserving the original spatial structure of the data points.",
}
```

[DirectProbe]: https://aclanthology.org/2021.naacl-main.401/
[DirectProbeCode]: https://github.com/utahnlp/DirectProbe
[data_url]: https://drive.google.com/drive/folders/1mlF-O20Zsa_jJG3tjV-vVrIivY71_R5P
