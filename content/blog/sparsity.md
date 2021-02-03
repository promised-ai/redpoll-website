+++
title = "The Challenges of Missing Data and Sparsity"
date = 2021-01-06
template = "post.html"
draft = false

[taxonomies]
categories = ["data science", "demo", "ai"]

[extra]
author = "Bryan Dannowitz"
subheading = "And how Redpoll's Reformer handles it in stride"
image = "sparsity/flashlight.jpg"
theme = "light-transparent"
+++

# Intro

Incomplete datasets are a fact of life in almost every discipline. All
but the most meticulously curated datasets are likely to have some degree
of missing information. How does one handle modeling when these gaps are present?
How does this scale with increasing sparsity of data?

In this post:

* **How is sparsity usually handled?**
* **How Redpoll's Reformer engine makes predictions with sparse data**
* **Reformer's data-aware ability to impute all missing values**
* **How imputation/prediction performance can vary as sparsity increases**

# Filling In the Gaps

There are a handful of strategies out there to handle missing values. Here
is a (certainly not exhaustive) list of such methods:

* Case deletion, or the removal or records that have missing values
* Fill with sentinel values (e.g. -1, -9999, "missing")
* Mean / median / mode substitution
* Regression (or classification) imputation
* Fill forward (for time-series-like observations)

Many of these act as stop gaps that simply allow one to be able to move
on to the next step of the modeling process. These can pollute
that very process since models can't tell real values from missing-and-filled
ones. The regression-filling approach is likely the most effective, but this
becomes intractable with highly sparse data.

Wouldn't it be better for a predictive model to be able to grasp
the concept: _"we don't know what all the attributes are, and that's okay"_?

# Reformer: Built to handle incomplete data

Customer data is often plagued
with incomplete information. In this example, we will use the mercifully complete
[Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
dataset and induce some artificial sparsity. A certain proportion (10%, for
starters) of the information will be dropped "Missing Completely at Random"
(MCAR). With this, we will show Reformer making predictions of the target (`Churn`).

First we apply the MCAR process to the Telco Churn dataset, dropping 10% of
_all_ feature values. We can visualize the degree of the absences as well:
```python
import pandas as pd
import missingno as mno

# Load raw data that is 25% unlabeled
df = pd.read_csv("telco_churn.csv", index_col="id")

# A helper function that drops 10% of all feature values
missing_df = missing_completely_at_random(df, frac=0.1, target="Churn")

mno.matrix(missing_df.sample(250))
```
![Missing Data Visualization](img/sparsity/mno.png)

Here, the black bars indicate where we have data, and the white gaps
between them represent missing values. In this case, 10% of feature
data is missing, and 25% of samples are unlabeled (have no `Churn` value).

Now we run the Reformer engine on this new dataset and impute all missing
target values.

Two things to note in this example:
* There is **no requirement to fill anything in or take any special
preprocessing steps** for training and imputation to work
* One does not need to specify which samples we need imputation on.
Reformer will provide a prediction for each one that is missing
a `Churn` value.

```python
from redpoll import reformer

# Connect to the Reformer server which is running locally on this file
client = reformer.Client("0.0.0.0:8000")

# Impute the missing values we care about
y_pred = client.impute("Churn")
y_pred
```
```
id              uncertainty     Churn
7590-VHVEG	0.012740	Yes
9305-CDSKC	0.007911	Yes
0280-XJGEX	0.093614	No
5129-JLPIS	0.002760	No
3655-SNQYZ	0.004449	No
...	...	...
9710-NJERN	0.009755	No
7203-OYKCT	0.004449	No
8456-QDAVC	0.015933	Yes
2234-XADUH	0.004449	No
3186-AJIEK	0.004449	No

1761 rows × 2 columns
```

These values can be compared against the true values by whatever evaluation
metric desired.

## Impute Every Feature

Now, we have a dataset in which 10% of the features values are missing.
Reformer, without any additional configuration or commmands, stands ready
to impute **any other feature of any type** that exists in the dataset.
This applies to both categorical and numeric imputation tasks.

```python
client.impute("MonthlyCharges")
```
```
id              uncertainty	MonthlyCharges
7590-VHVEG	0.285218	29.506515
7892-POOKP	0.004120	90.725902
7469-LKBCI	0.001955	19.962660
0280-XJGEX	0.110770	91.588106
4183-MYFRB	0.126903	87.122360
...	...	...
7203-OYKCT	0.029650	98.639224
9281-CEDRU	0.239033	67.092866
2234-XADUH	0.029650	98.639224
4801-JZAZL	0.168539	35.743668
8361-LTMKD	0.017313	68.429939

704 rows × 2 columns
```
This stands apart from the current paradigm of operational data science:
_one must always decide the question we want to ask before modeling._
Reformer simplifies that _modus operandi_ and lets you ask any question you want
**without any retraining or reconfiguration.**

\{\{ jsplot(path="static/img/sparsity/reformer_rfc.html", img="/img/sparsity/reformer-accuracy.png", caption="test") }}

# How much sparsity can Reformer handle?

Since we're introducing sparsity artificially, we can take this to the extreme
and explore how Reformer performs on increasingly sparsified data. To do this,
we'll create datasets that are missing 20% missing, 30% missing, on up to 90%
of all feature data missing. Then, it's a matter of:

1. Feed these individual datasets into the Reformer engine
1. Pick a feature, and impute its missing values
1. Refer to the complete dataset and evaluate the imputation
1. Repeat for all features
1. Visualize individual performance as a function of sparsity

<div class="center">
    <embed src="/img/sparsity/reformer_telco_accuracy.html" width="800" height="620"> </embed>
</div>

Performance does indeed decrease as sparsity increases (as it should when data is lost),
but the takeaway here is: **Reformer can handle any degree of sparsity and deliver
without any preprocessing**.

Let's go ahead and compare Reformer to a stock classifier for the `Churn` target
prediction at varying levels of sparsity. In order to do so with something like a
Random Forest Classifier (RFC), we will need to perform the following preprocessing:

1. Calculate the modes (categorical) and means (numeric) of all features
1. Fill all missing values with mode/mean values, since RFC cannot handle missing info
1. Encode all categorical features into numeric/boolean types (here, one-hot encoded)
1. Train the RFC, specifying the feature set and the target at training

<div class="center">
    <embed src="/img/sparsity/reformer_rfc.html" width="800" height="620"> </embed>
</div>




# Conclusion

Here, we have exhibited the unmatched flexibility of Reformer to model **the whole dataset** as
it is, without additional steps to make the metaphorical square peg fit into the round hole.

As the sparsity increased in the above example, the dataset began to resemble a whole different
brand of dataset — an unstructured recommender-engine style dataset. In an upcoming post, we will
show how the same Reformer engine, without any changes in configuration, could be used to model
even these kinds of issues with the _MovieLens 100k_ dataset.

If you would like to hear more about how the Reformer engine can transform and simplify your analytical, descriptive, and predictive pipelines, reach out to us via our Contact link below.
