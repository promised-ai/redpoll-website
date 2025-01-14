+++
title = "Plover Found 9 Errors in the UC Irvine AI4I Predictive Maintenance Dataset"
date = 2024-11-20
template = "post.html"
draft = false

[taxonomies]
categories = ["data science", "data quality", "plover"]

[extra]
author = "Baxter Eaves"
subheading = "Finding errors in the code behind the synthetic data"
image = "auto-shop-1.jpg"
theme = "light-transparent"
front_page = false
+++

**TL;DR**: Plover found nine mislabeled records. Download the cleaned data [here](/datasets/#uci-ml-repository-ai4i-2020-predictive-maintenance-dataset). Try Plover in your browser [here](/plover).

---

I believe most, if not all, of the people I know who work with data would agree that all real-world datasets of substance have errors. But under what conditions could (or should) a datasets not have errors? In the machine learning space, one way we get around the messiness and sparsity of real-world data is by building computer programs to programatically generate synthetic data. You may think "surely data generated by a program would be, by definition, error free, right?" Wrong. 

There is a subtle distinction that often gets overlooked by the data quality community. Your data are true. No matter how "bad" they are. The data were generated by the true data generating process and ended up in your database. So-called "erroneous" data isn't in fact erroneous but is evidence of an error in the data process. Maybe someone fat-fingered something, maybe units were not normalized across sources. In the latter case, the erroneous data actually reflect erroneous code.

Here we'll show how we used [Plover](/plover) to identify errors in (the code that generated) a highly curated synthetic dataset in one of the most used repositories for clean machine-learning-ready data: the [UCI Machine Learning Repository AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

To quote the UCI repository page

> The AI4I 2020 Predictive Maintenance Dataset is a synthetic dataset that reflects real predictive maintenance data encountered in industry.

> The dataset consists of 10 000 data points stored as rows with 14 features in columns
> UID: unique identifier ranging from 1 to 10000

- product ID: consisting of a letter L, M, or H for low (50% of all products), medium (30%) and high (20%) as product quality variants and a variant-specific serial number
- air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
- process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
- rotational speed [rpm]: calculated from a power of 2860 W, overlaid with a normally distributed noise
- torque [Nm]: torque values are normally distributed around 40 Nm with a Ïƒ = 10 Nm and no negative values. 
- tool wear [min]: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. and a
- 'machine failure' label that indicates, whether the machine has failed in this particular datapoint for any of the following failure modes are true.

The machine failure consists of five independent failure modes

- tool wear failure (TWF): the tool will be replaced of fail at a randomly selected tool wear time between 200 and 240 mins (120 times in our dataset). At this point in time, the tool is replaced 69 times, and fails 51 times (randomly assigned).
- heat dissipation failure (HDF): heat dissipation causes a process failure, if the difference between air- and process temperature is below 8.6 K and the tool's rotational speed is below 1380 rpm. This is the case for 115 data points.
- power failure (PWF): the product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails, which is the case 95 times in our dataset.
- overstrain failure (OSF): if the product of tool wear and torque exceeds 11,000 minNm for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain. This is true for 98 datapoints.
- random failures (RNF): each process has a chance of 0,1 % to fail regardless of its process parameters. This is the case for only 5 datapoints, less than could be expected for 10,000 datapoints in our dataset.

Let's dig in. We're going to do everything in the python bindings today, so we'll start by creating a local Plover instance.

```python
from plover import Plover
from plover.source import DataFrame

plvr = (
    Plover.local(
        source=DataFrame.csv(
            "ai4i2020.csv",
            index_col="UID",
            schema="auto"
        ),
        store="ai4i2020.plvrstore",
    )
    .fit()
    .compute_metrics()
    .metalearn()
    .persist()
)
```

We use a local dataset using the `DataFrame` source, store our metadata locally using the `Local` store, and use a local machine learning backend. The `fit` command builds an inference model of the data. The `compute_metrics` method computes the error/anomaly metrics for each cell. The `metalearn` method creates a second-order machine learner that allows similarity queries, as we'll see below. The `persist` method saves everything to the local store so we can resume later.

Now that we've done all that, let's find the top five most likely errors.

```python
plvr.errors(top=5)
```

| row | col | ic | obs | pred |
|-----|-----|----|-----|------|
| 9016 | TWF | 142.939812 | 0 | 0 |
| 5537 | HDF | 115.492924 | 0 | 1 |
| 9016 | OSF | 113.370143 | 0 | 1 |
| 4703 | PWF | 75.064483 | 0 | 1 |
| 1493 | OSF | 50.205087 | 0 | 0 |


The above table shows the inconsistency metric as `ic`. The important thing is that more inconsistency means more likely an error. You can see the steep falloff in `ic` from top to bottom. Interestingly there are two cells from record `9016` in the top five errors. Let's ask plover to explain which features are responsible for `TWF`'s high inconsistency on record `9016`.

```python
plvr.explain(row="9016", col="TWF")
```

|    | feature                |   ic     | observed | predicted |
|----|------------------------|----------|----------|-----------|
|  0 |                        | 142.94   |          |           |
|  1 | Machine failure        | 5.06351  | 1        | 0         |
|  2 | Tool wear [min]        | 0.549236 | 210.0    | 114.47    |

The above tables shows us how much inconsistency is left after removing certain features. Features are sorted by their contribution to the uncertainty in the target variable, in this case, `TWF`.
Plover is telling us that the `Machine Failure` value is generally responsible for all the inconsistency. In a distant second is `Tool wear [min]`. Let's take a look at record "9016".

```python
plvr.data(row="9016")
```

|                         | 9016   |
|:------------------------|:-------|
| Type                    | L      |
| Air temperature [K]     | 297.2  |
| Process temperature [K] | 308.1  |
| Rotational speed [rpm]  | 1431.0 |
| Torque [Nm]             | 49.7   |
| Tool wear [min]         | 210.0  |
| Machine failure         | 1      |
| TWF                     | 0      |
| HDF                     | 0      |
| PWF                     | 0      |
| OSF                     | 0      |
| RNF                     | 0      |

Right off the bat we see that `Machine failure` is `1` but all the failure modes are `0`. According to the data documentation

> If at least one of the above failure modes is true, the process fails and the 'machine failure' label is set to 1. 

We have an error! Plover correctly identified that `Machine Failure` should be 0.

An error that happens once can happen again, so we need to find similar errors. We can do this one of two ways: we can write a rule and filter the dataset (which would be really easy in this case, but really difficult in general), or we can ask plover to find similar cells.

```python
plvr.similar_cells(row="9016", col="Machine failure").head(10)
```

|   row |      similarity |
|------:|---------:|
|  4045 | 0.789062 |
|  5942 | 0.789062 |
|  4685 | 0.789062 |
|  1438 | 0.773438 |
|  5537 | 0.773438 |
|  2750 | 0.742188 |
|  6479 | 0.71875  |
|  8507 | 0.71875  |
|  5910 | 0.5625   |
|  4703 | 0.554688 |

There are eight cells that have a high meta similarity with the error we found. Since it's easy to do in this case, let's filter the data to pull out all the rows in which `Machine failure` is 1 but every failure mode is 0.

```python
failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
df = plvr.df()
df[ 
    (df[failure_modes].sum(axis=1) == 0) 
    & 
    (df['Machine failure'] > 0)
][failure_modes + ["Machine failure"]]
```

|  row |   TWF |   HDF |   PWF |   OSF |   RNF |   Machine failure |
|-----:|------:|------:|------:|------:|------:|------------------:|
| 1438 |     0 |     0 |     0 |     0 |     0 |                 1 |
| 2750 |     0 |     0 |     0 |     0 |     0 |                 1 |
| 4045 |     0 |     0 |     0 |     0 |     0 |                 1 |
| 4685 |     0 |     0 |     0 |     0 |     0 |                 1 |
| 5537 |     0 |     0 |     0 |     0 |     0 |                 1 |
| 5942 |     0 |     0 |     0 |     0 |     0 |                 1 |
| 6479 |     0 |     0 |     0 |     0 |     0 |                 1 |
| 8507 |     0 |     0 |     0 |     0 |     0 |                 1 |
| 9016 |     0 |     0 |     0 |     0 |     0 |                 1 |

Cross-checking the meta-similar records with the filter, we see that meta-similarity found the same entries as the hard-coded rule. Nice!

Digging in more, and recreating the failure rules supplied by the dataset authors, it turns out the entry `9016` could have suffered from a tool wear failure (TWF) since its tool wear was greater than 200 minutes. However, since `TWF` is a random and rare failure for tool wears greater than 200, we decided to mark `9016` as not having failed in the cleaned dataset.

## Conclusion

All processes are prone to errors, so all data &mdash; even synthetic &mdash; can contain the evidence of those errors in the form of so-called bad data. We showed how Plover can easily identify erroneous data and how we can use meta-similarity to identify similar errors. In this case, Plover found a very salient error in part `9016` and used meta-similarity to find all other data exhibiting that particular error without having to write a single rule.

## Changelog

Download the cleaned data [here](/datasets/#uci-ml-repository-ai4i-2020-predictive-maintenance-dataset). And try Plover in your browser [here](/plover).

| UID (row) |          Column | Original | New |
|-----------|-----------------|----------|-----|
|      9016 | Machine Failure |        1 |    0|
|      5537 | Machine failure |        1 |    0|
|      1438 | Machine failure |        1 |    0|
|      2750 | Machine failure |        1 |    0|
|      4045 | Machine failure |        1 |    0|
|      4685 | Machine failure |        1 |    0|
|      5942 | Machine failure |        1 |    0|
|      6479 | Machine failure |        1 |    0|
|      8507 | Machine failure |        1 |    0|

