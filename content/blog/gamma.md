+++
title = "Use Case: Gamma Ray Detection with the MAGIC Telescope"
date = 2020-12-05
template = "post.html"
draft = false

[taxonomies]
categories = ["data science", "demo", "ai"]

[extra]
author = "Bryan Dannowitz"
subheading = "Hands-on with Reformer - Uncertainty and Anomaly Detection"
image = "magic-laser-calibration.jpg"
theme = "light-transparent"
+++

Machine Learning is often spoken of in hushed, reverent tones as a panacea to
whatever data woes one might have. This is in unfortunate contrast to the
capabilities of the typical solutions that are widely available. Most
any model — from the humble perceptron to the formidable, pre-trained VGG-19
deep neural network — are very focused and limited in their scope; they're
trained and tuned to perform one task well. They tend to be of little use for
any other task, even one which is based on the exact same dataset.

In this post, we'll review a simple supervised classification task as it's
commonly encountered, and deal with it via traditional ML methods. Then we
will extend the scope of what can be performed on such a dataset when
a more holistic AI engine is applied. Topic emphasized will be inference
(predictions), simulation, anomaly detection, and uncertainty estimation.

# Gamma Ray Detection with the MAGIC Observatory

In the Canary Islands, there is an observatory consisting of two 17m telescopes
whose primary purpose is to detect traces of rare high-energy photons,
known as gamma rays. These extra-solar visitors bombard the Earth from
fascinating, catastrophic origins. Studying these events (incident angle, energy)
can yield insights into the science of faraway cataclysmic events and
large astronomical objects.

The problem is that Earth is continuously pummeled by high-energy particles
(cosmic rays) that are decidedly _not_ gamma rays, causing a noisy
background of gamma ray-like events. Being able to automatically discriminate
between real gamma shower events and _"hadronic showers"_ would be very useful,
especially since hadronic showers much more commonly occur.

![Measurements of an incident gamma/hadron shower](/img/magic-experiment.png)

Without going into much detail, these events light up "pixels" of a detector,
forming an ellipse. These ellipses can be parametrized by characteristics such
as _distance from the center of the camera_ (`fDist`), length (`fLength`) and
width (`fWidth`) of the ellipse, and angle of its long axis (`fAlpha`). There
are also metrics with respect to how bright the pixels are (`fSize`) and how
concentrated that brightness is (`fConc`). You can find the dataset and the
full definition of the features at the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope)

```python
import pandas as pd
import seaborn as sns

df = pd.read_csv("magic04.data", index_col=0)
sns.pairplot(
    data=df,
    vars=["fLength", "fWidth", "fConc", "fAlpha", "fM3Trans"],
    hue="class",
    kind="kde",
    corner=True,
)
```

![MAGIC Pairplot](/img/gamma/magic-pairplot.png)

From this brief peek into the features broken down by target class ('g' for
gamma, 'h' for hadron), one can see significan distribution overlap, which
means that gamma detection will not be too easy, and there _should_ be some
uncertainty inherent in many predictions. Let's start by treating this as a
typical binary classification problem.

# Traditional ML Approach

What can a widely available, off-the-shelf solution do with this simple, but
challenging dataset?

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

X = df.drop(["class"], axis=1)
y = df["class"].replace({"g": 1, "h": 0})

rfc = RandomForestClassifier()
metrics = cross_validate(
    estimator=rfc,
    X=X,
    y=y,
    cv=5,
    scoring=["accuracy", "precision", "roc_auc", "average_precision"]
)
for m in metrics:
    print(f"{m}: {round(np.mean(metrics[m]), 3)} ± "
          f"{round(np.std(metrics[m]), 3)}")

# fit_time: 3.48 ± 0.068
# score_time: 0.114 ± 0.007
# test_accuracy: 0.88 ± 0.004
# test_precision: 0.885 ± 0.006
# test_roc_auc: 0.935 ± 0.005
# test_average_precision: 0.957 ± 0.004
```

Here we have instantiated and trained a tool for one specific task.
It faithfully measures up with an excellent 89% precision — a good metric
to focus on for a situation where you want to end up with a pure gamma sample
to study.

But... what else can solutions like this accomplish? Let's give it a fair
shake and  cover feature importance, which can be forcibly extracted from a
trained model. Without writing an entire tangent on caveats, cautions, and
co-linearities, we can demonstrate the ability to derive impurity- and
permutation-based feature importances.

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# Split the data and train a single model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=314,
)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# "Default" impurity-based feature importances
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Permutation importance, which shuffles all the values of
# a single feature to measure the drop in performance
pi = permutation_importance(
    estimator=rfc,
    X=X,
    y=y,
    scoring="average_precision"
)
pi_indices = np.argsort(pi["importances_mean"])[::-1]

# Plot two kinds of feature importance measures
# TODO: Leave this verbose plotting section out?
fig, axs = plt.subplots(1, 2, figsize=(18, 5))

axs[0].set_title("Impurity-Based Feature Imporances")
axs[0].bar(
    x=range(X.shape[1]),
    height=importances[indices],
    yerr=std[indices],
    align="center"
)
axs[0].set_xticks(range(X.shape[1]))
axs[0].set_xticklabels(X.columns[indices], rotation=40)
axs[0].set_xlim([-1, X.shape[1]])

axs[1].set_title("Permutation-Based Feature Importances")
axs[1].bar(
    x=range(X.shape[1]),
    height=pi["importances_mean"][pi_indices],
    yerr=pi["importances_std"][pi_indices],
    align="center"
)
axs[1].set_xticks(range(X.shape[1]))
axs[1].set_xticklabels(X.columns[pi_indices], rotation=40)
axs[1].set_xlim([-1, X.shape[1]])
plt.show()
```

![RFC Impurity and Permutation Importances](/img/gamma/gamma-rfc-importance.png)

This type of inspection can lead to a better understanding of what
drives the predictions made, and can better inform the user on
certain matters specific to the problem at hand.

There's also "partial dependence", which can be helpful in the same way.

```python
from sklearn.inspection import plot_partial_dependence

# The model can be exploited by changing individual samples' individual
# features to observe the average change in prediction values
fig, ax = plt.subplots(figsize=(15, 7))
plot_partial_dependence(
    rfc, X=X.sample(100), features=X.columns, n_cols=5, ax=ax
)
fig.tight_layout()
```

![RFC Partial Dependence Plots](/img/gamma/rfc-partial-dependence.png)

There are some great insights in plots like this that can tell you
"if this value is higher, then prediction probabilities of class 'g'
drop," as is the case with `fAlpha` here.

And there we have it, a certainly not exhaustive, but a fairly
comprehensive assessment of what can do with a traditional ML model
on a dataset. One is required to specify the feature set and the target,
train the model, and then one is able to:

1. Make predictions
1. Evaluate the predictions
1. Exploit the model to learn about how the features influence the prediction

_... and that's about it_. The problem is that this approach assumes:

* **You will only care about the answer to this one question**
* **You don't care a whole lot about the input features**
* **You don't want any information pertaining to individual samples**
* **All of your data is anomaly-free, or you don't care about anomalous or
surprising values**

This sets aside the matter of being able to update the model as new data
comes in, as there are architectures our there that can support live model
updates. Now, there are certainly platforms out there that
go the full Dr. Frankenstein and stitch together N different individual models
to be able to describe each individual feature or do anomaly detection,
but this becomes inelegant and unsatisfying.

Now that the lede has been sufficiently buried, let's take a look at what can
be achieved if we take a step into a new world of holistically data-aware
modeling like we find in Redpoll's Braid Engine.

# Redpoll's Braid Engine

The Braid engine runs as a server instance to which one can connect and make
any number of requests. This has a number of benefits over maintaining,
updating, and distributing a model file, but let's focus on task utility
for the moment.

Let's begin with inference — roughly the equivalent of _prediction_ in the
traditional ML sense. We have prepared a data file with the 1/3rd of the
`class` values `NaN`'d out; the same values as in the `X_test` samples above, for
_apples-to-apples_ comparison's sake. We then created a _braid file_ from
that data and started up a server on it locally. That done, inference, along
with many other operations, are a simple matter to execute.

```python
import pybraid

# Create a client instance connected with the server
c = pybraid.Client("0.0.0.0:8000", secure=False)

# Identify which rows you want a prediction for
pred_ixs = X_test.index.values.astype(str)

# Request imputed values for these rows
y_pred_rp = c.impute(col="class", rows=pred_ixs)
y_pred_rp.head()

#        uncertainty class
# 18597     0.019604     h
# 16776     0.153241     g
# 15694     0.041106     g
# 16397     0.020332     h
# 3772      0.201858     g
```

Here's our first departure from what one might be used to seeing. Braid, by
default, pairs its individual inferences along with an uncertainty value.
More on this later.

The evaluation of this set of predictions yields:

```python
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix

accuracy = accuracy_score(y_test, y_pred_rp)
precision = precision_score(y_test, y_pred_rp)

def get_conf_df(y_test, y_pred):
    """Helper function to render a nice confusion matrix dataframe."""
    conf_arr = confusion_matrix(y_test, y_pred)
    conf_df = pd.DataFrame(
        {
            "Predicted Negative": [
                conf_arr[0][0],
                conf_arr[1][0],
                conf_arr[0][0] + conf_arr[1][0],
            ],
            "Predicted Positive": [
                conf_arr[0][1],
                conf_arr[1][1],
                conf_arr[0][1] + conf_arr[1][1],
            ],
            "Total": [conf_arr[0].sum(), conf_arr[1].sum(), conf_arr.sum()],
        }
    )
    conf_df.index = pd.MultiIndex.from_tuples(
        [("Actual", "-"), ("Actual", "+"), ("Total", "")]
    )
    conf_df.columns = pd.MultiIndex.from_tuples(
        [("Predicted", "-"), ("Predicted", "+"), ("Total", "")]
    )
    conf_df.style.set_properties(**{"text-alig": "right"})
    return conf_df

print(f"Test Accuracy: {round(accuracy, 3)}")
print(f"Test Precision: {round(precision, 3)}")

conf_df = get_conf_df(y_test, y_pred_rp)
conf_df

# Test Accuracy: 0.85
# Test Precision: 0.86
#
#          Predicted       Total
#                  -     +
# Actual -      1605   606  2211
#        +       335  3731  4066
# Total         1940  4337  6277
```

Feature importance can also be derived from the braid engine by calculating
the proportion of information related to the target carried in each column.

```python
def get_feat_importance(client, target_cols, predictor_cols):
    """Calculate the information proportion for each predictor."""
    info_props = np.array(
        [
            client.info_prop(
                target_cols=target_cols,
                predictor_cols=[predictor_cols[i]],
            )
            for i in range(len(predictor_cols))
        ]
    )
    ip_data = (
        pd.DataFrame({
            "Predictor": predictor_cols,
            "Information Proportion": info_props}
        )
        .sort_values("Information Proportion", ascending=False)
        .reset_index(drop=True)
    )
    return ip_data

ip_data = get_feat_importance(
    client=c,
    target_cols=["class"],
    predictor_cols=[col for col in df.columns if col != "class"]
)

fig = plt.figure(figsize=(12, 8))
sns.barplot(
    data=ip_data,
    x="Predictor",
    y="Information Proportion",
    color="#333333",
)
```

![Braid Info Proportion](/img/gamma/info-prop.png)

We see in this apples-to-apples evaluation that a stock RandomForestClassifier
wins out, 88.5% to 86% on precision. If one is not participating in a Kaggle
competition, eking out every point possible, and is able to accept this
marginal difference, we can begin to delve into the opportunities created with
Redpoll's Braid engine.

# Beyond Target Prediction

Let's turn our attention to how the Braid engine has modeled the **entire**
dataset. Let us consider a situation where a gamma ray has hit our hard disk
and we have lost 1/3rd of the values in our dataset.

```python
missing_df = df.copy()

# Get the number of feature values in the dataset
vals = missing_df.drop(["class"], axis=1).values.flatten()

# Select one third of them randomly
null_ixs = np.random.choice(
    range(len(vals)),
    size=round(len(vals)/3.0),
    replace=False
)
vals[null_ixs] = np.NaN

# Overwrite the values in the dataframe with this 1/3rd empty set
missing_df.iloc[:, :-1] = vals.reshape(missing_df.shape[0], missing_df.shape[1] - 1)
missing_df[["fLength", "fWidth", "fConc", "fAlpha"]].sample(5)

#        fLength  fWidth   fConc   fAlpha
# id
# 11327  19.4348  7.3353  0.7744  60.2180
# 15902      NaN     NaN  0.3250      NaN
# 18111  52.5016  9.5828  0.3645  32.6580
# 9525   20.1855     NaN     NaN      NaN
# 18742  20.8995     NaN  0.7428  81.0387

import missingno as mno
mno.matrix(missing_df)
```

![Dataset with aggressive data dropout](/img/gamma/missing.png)

In this case of aggressive data loss, most traditional methods would have
issues with using a dataset like this for its class prediction. One might
handle this by using mean/median imputation on the missing values in their
data pre-processing step, or they might even go overboard and train individual
models for predicting each feature.

Braid, however, performs just as it did before without any issue. It is able
to use what is given in order to render a prediction (along with uncertainty).

```python
y_pred = c.impute("class")

# Discard uncertainty value for now
y_pred = y_pred["class"]
y_pred = y_pred.replace({"g": 1, "h": 0})

print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 3)}")
print(f"Precision: {round(precision_score(y_test, y_pred), 3)}")
get_conf_df(y_test, y_pred)

# Accuracy: 0.815
# Precision: 0.813
#
#          Predicted       Total
#                  -     +
# Actual -      1343   868  2211
#        +       296  3770  4066
# Total         1639  4638  6277
```

But, to go even further, imagine you did want to fill in the gaps as best you
can. Go ahead and use your Braid engine to make predictions on
***any feature*** in your dataset.

```python
from sklearn.metrics import r2_score

# Predict any missing values of any feature
y_pred_alpha = c.impute("fAlpha")["fAlpha"]

# Pull the true values from the full dataset
y_true_alpha = df.loc[y_pred_alpha.index.astype(int), "fAlpha"]

# Do some evaluation
print(f"R2: {round(r2_score(y_true_alpha, y_pred_alpha), 3)}")

# R2: 0.06

# Plot the residuals
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(y_pred_alpha.values - y_true_alpha.values, ax=ax)
_ = ax.set_title("fAlpha Prediction Residuals")
```

![fAlpha Residuals](/img/gamma/falpha-residuals.png)

It bears repeating: Braid has modeled the **entire dataset**. As such, it is able
to model all of the features, not just the designated target. In fact,
Braid doesn't even ask you to designate a target when providing it data
to work with.

# Data-Aware

It's often the case that once our data is fed into fitting a model, we're done
looking at it. It is the culmination of all the pre-screening and preprocessing
that has been laid out before. But what if your framework could help you be
introspective about the information that you have.

## Similarity

With Braid, one can look to a single sample and find the most similar other
samples in your dataset. This is no mere euclidean or cosine distance
calculation. Those metrics are affected by scale. More importantly, what happens
if an attribute's distribution is flat noise across the feature space? In that
case, different values shouldn't matter so much with sample similarity. Simple
distances cannot suffice when identifying similar samples.

```python
df.iloc[[0, 26, 15000]][["fLength", "fWidth", "fConc", "fAlpha", "class"]]

#        fLength   fWidth   fConc   fAlpha class
# id
# 0      28.7967  16.0021  0.3918  40.0920     g
# 26     27.2304  19.2817  0.3710  77.5379     g
# 15000  37.9753   5.1561  0.6779  14.3465     h

# Retrieve the similarity between pairs of rows
c.rowsim([["0", "26"], ["0", "15000"], ["26", "15000"]])

#     A      B  rowsim
# 0   0     26   0.625
# 1   0  15000   0.000
# 2  26  15000   0.000

# Find the most similar rows of all pairs in the dataset
all_pairs = [["0", str(i)] for i in range(1, len(df))]
(c.rowsim(all_pairs)
     .sort_values("rowsim", ascending=False)
     .head())

#        A      B  rowsim
# 4421   0   4422  1.0000
# 6516   0   6517  0.9375
# 11020  0  11021  0.9375
# 2396   0   2397  0.9375
# 657    0    658  0.8750

# Same, but asking for similar rows *with respect to* one or more features
(c.rowsim(all_pairs, wrt=["fAlpha"])
     .sort_values("rowsim", ascending=False)
     .head())

#       A     B  rowsim
# 2396  0  2397   1.000
# 4421  0  4422   1.000
# 5430  0  5431   0.875
# 5281  0  5282   0.875
# 7864  0  7865   0.875
```

## Anomaly Detection

Another capability with high utility is the ability to understand when a value
is surprising, or anomalous. Much like the similarity metric, this can be a
complicated measure to define. Yes, if a value is many sigma outside its
distribution, it's anomalous, but it's also anomalous if a gamma event has
a high `fAlpha` value (see jointplot above).

Traditional solutions stand helpless to identify these for you before
or after training. There are plenty of standalone solutions and models
that are made specifically for anomaly detection, but then you've just
increased the amount of complexity of your project.

Braid can natively highlight these samples for your consideration and evaluation.

```python
from math import ceil, floor

# Calculate the log probability of each sample
logps = c.logp_scaled(
    cols=df.columns,
    values=[list(x) for x in df.values]
)
# Convert to probability
sample_probs = [np.exp(x) for x in logps]

# Get the least probable sample
anomaly_ix = np.argmin(sample_probs)
anomaly = df.iloc[anomaly_ix]

# Plot the values of this sample over the base distributions
fig, axs = plt.subplots(2, 5, figsize=(15, 5))
for i, col in enumerate(df.columns[:-1]):
    ax = axs[floor(i // 5)][i%5]
    sns.histplot(df[col], ax=ax, kde=True)
    ax.vlines(anomaly[col], 0, ax.get_ylim()[1], color='r')
fig.tight_layout()
```

![Anomalous Sample](/img/gamma/anomalous-sample.png)

We can see that, in most feature distributions, this sample is far into the
tails. If acquainted with the domain at hand, this sample (and others up to a
certain probabilistic level) can be evaluated to be either real or erroneous.
Whether or not one might wish to keep an erroneous outlier in the training set
is a judgment call, but one that can only be made once outliers have been
identified.

The capabilities and data-awareness doesn't end there; it is possible to find
samples that are anomalous with respect to a _single feature_ or set of features.
There may be cases where most of a sample's values are well within the realm of
high probability, but feature X might be way off from what would be expected.
This can be especially surprising given that the rest of the values are
being rather average.

```python
# Calculate all the fWidth surprisals and get the top 5 most surprising
(c.surprisal("fWidth")
    .sort_values(by=["surprisal"], ascending=False)
    .head())

#          fWidth  surprisal
# 17717  256.3820  12.352112
# 17754  228.0385  10.388123
# 17619  220.5144   9.477771
# 17821  112.2661   8.524242
# 13496  201.3640   8.389742

# Inspect the most surprising value w.r.t. the whole distribution
ax = sns.histplot(df.fWidth, kde=True)
_ = ax.vlines(256.382, 0, ax.get_ylim()[1], color='r')
```

This is a trivial example of an outlier. It's the same sample as the one
found above, and can be seen in the `fWidth` plot.  This is not too
interesting, but consider this example:

```python
# Calculate all the fAlpha surprisals and get the top 5 most surprising
(c.surprisal("fAlpha")
    .sort_values(by=["surprisal"], ascending=False)
    .head())

#         fAlpha  surprisal
# 12282  12.4080   8.358472
# 4462   89.9155   6.764617
# 11103  12.6281   6.717841
# 18038  89.7370   6.661253
# 12450  89.4816   6.628395

ax = sns.histplot(df.fAlpha, kde=True)
_ = ax.vlines(12.6281, ymin=0, ymax=ax.get_ylim()[1])
```
![fAlpha Anomaly](/img/gamma/alpha.png)

Here we have an interesting case where a particularly common `fAlpha` value
of 12.4 is even more surprising than a fringe value of 89.9 — why is that?
This is due to the context of the rest of the sample's information. We can
get even more introspective about this anomaly by utilizing Braid's
**simulation** capabilities.

## Simulation

While the MAGIC gamma ray dataset is itself a Monte Carlo generated sample,
by training the Braid engine with its data, Braid itself can now act as a
mimicking Monte Carlo generator. The MAGIC dataset is imbalanced, favoring
gamma ray samples over hadronic 2-to-1. If a more balanced set is desired,
downsampling is an option, or selecting the same hadron samples over again
is also possible.

But what if we generate our own simulated hadron samples? With `simulate()`,
it's possible to create new samples based on what we know about the features,
based on how they depend and relate to each other. Certain values can be hard
set, such as the `class` value.

```python
# Generate new MC hadron samples
c.simulate(df.columns[:-1], given={"class": "h"}, n=5)

#      fLength     fWidth     fSize  ...    fM3Trans     fAlpha       fDist
# 0  94.632702  52.056796  3.673992  ...    9.237673  11.185377  137.102271
# 1  17.655791  10.795115  2.537168  ...   -5.829544  65.106615  134.789055
# 2  17.635636  73.751666  3.679984  ...  196.103052  52.722776  258.497931
# 3  45.758857   7.080488  2.647146  ...   -2.956160  65.282567  150.788139
# 4  43.598523  25.939003  2.922657  ...   13.998317  -0.488088  134.966457
```

Multiple `given`s can be assigned, allowing the user to simulate any feature
given any hypothetical situation.

```python
sim = c.simulate(
    cols=["fDist", "fConc", "class"],
    given={
        "fAlpha": 2.5,
        "fLength": 140,
    },
    n=500
)
sim.head()

#       fDist     fConc class
# 0  307.080782  0.243650     g
# 1  246.520126  0.329555     g
# 2  282.019657  0.188556     g
# 3  307.293975  0.231327     h
# 4  342.755502  0.225058     h

With this capability in hand, let's return to that anomalous `fAlpha = 12.4`.
We can simulate a distribution of `fAlpha` based on the rest of the values in
that particular sample.

```python
# Get the non-fAlpha values from this sample
falpha_sample_data = df.loc[12282].drop("fAlpha").to_dict()

# Make 10k simulated values and plot the distribution
sim_alphas = c.simulate(
    cols=["fAlpha"],
    given=falpha_sample_data,
    n=10000,
)
sns.histplot(sim_alphas, kde=True)

# Plot the anomalous fAlpha value over the distribution
ax.vlines(12.4, ymin=0, ymax=ax.get_ylim()[1], color='r')

# Assuming gaussian-like distribution, plot std dev intervals
(mean, std) = (fa.mean(), fa.std())
ax.vlines(
    [mean-i*std for i in range(-3,4)],
    ymin=0,
    ymax=ax.get_ylim()[1]*0.1, color='gray',
)
```
![Simulated fAlpha](/img/gamma/sim-alpha.png)

And here we have the awaited answer to _"why is this perfectly average
`fAlpha` value considered anomalous?"_ Given the rest of the data in the
sample, according to Braid's understanding of how all the features
depend on and predict one another, `fAlpha` is expected to lie almost
entirely between 0 and 10. That it is 12.4 is pretty surprising to Braid,
as you can see it's past the 3σ mark of the right tail.

# Uncertainties

It's been touched on before, but let's talk about the ability to
provide. In a very similar procedure to the simulation method above,
we ask for a prediction given limited information:

```python
# Predict class, given two feature values
c.predict(
    "class",
    given={"fConc": 0.17, "fDist": 110.0}
)

# (Class prediction, uncertainty)
# ('g', 0.19497380732945924)
```

This uncertainty metric is not a measure of feature-specific variance,
It is instead a unitless metric specific to Redpoll's Braid
platform as a way for it to communicate its certainty in an imputation
or hypothetical prediction. This can be a very important metric to keep in mind
when making critical decisions. Perhaps, depending on the circumstances,
one might only want to take action when the certainty is high.

Let's give it some very clear hypotheticals and some confusing ones to
see what it does, using this jointplot as a reference.

![Jointplot for Uncertianty](/img/gamma/jointplot-uncertainty.png)

```python
# Values very likely to be a hadron shower, low uncertainty
c.predict(
    "class",
    given={"fAlpha": 75.0, "fM3Long": -100.0, "fWidth": 75.0}
)
# ('h', 0.015325633697621743)

# Values with high degree of overlap between gamma and hadron class,
# which will introduce some uncertainty
c.predict(
    "class",
    given={"fM3Long": 0.0, "fWidth": 10.0},
)
# ('g', 0.22257845777068186)

# Likely gamma fAlpha value, but a likely hadronic fM3Long value
# This contrast will increase the uncertainty of the prediction
c.predict(
    "class",
    given={"fAlpha": 1.0, "fM3Long": -75.0},
)
# ('g', 0.29104324615324395)
```

To provide a more intuitive understanding of what drives this these
predictions and uncertainties, let's assemble some visualizations.
First, it helps to understand that Braid has multiple internal _states_,
with each one providing its own probabilistic view on the question
posed. In general, if all of these states agree with each other,
the uncertainty will be low, but if they wildly differ, a superposition
of all of the distributions will provide a prediction, but the uncertainty
will be higher.


```python
def plot_internal_state_likelihoods(client, col, row_ix, value_list):
    """Use surprisal to visualize each state's prediction distribution.

    Parameters
    ----------
    client : pybraid.Client
    col : str
        Column to visualize the state distributions for.
    row_ix : str
        The index of the sample in the dataset to draw from
    value_list : list-like or array

    Returns
    -------
    matplotlib.pyplot.Axes

    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the likelihood distributions for each of braid's underlying states
    for state in range(client._nstates):
        surp = client.surprisal(
            col=col,
            rows=[row_ix] * len(value_list),
            values=value_list,
            state_ixs=[state],
        )
        # Convert to probability distribution
        surp["negexp_surprisal"] = np.exp(-surp["surprisal"])
        # Plot distribution, half transparent
        surp.plot(
            x=col,
            y="negexp_surprisal",
            ax=ax,
            c="gray",
            alpha=0.5,
        )
    # Calculate the surprisal over all internal states
    surp = client.surprisal(
        col=col,
        rows=[row_ix] * len(value_list),
        values=value_list,
        state_ixs=None,
    )
    surp["negexp_surprisal"] = np.exp(-surp["surprisal"])

    # Overlay this superposition
    surp.plot(x=col, y="negexp_surprisal", ax=ax, color="k")

    return ax


def uncertainty_plot(client, df, col, row_ix, value_list):
    ax = plot_internal_state_likelihoods(
        client=client,
        col=col,
        row_ix=row_ix,
        value_list=value_list
    )

    # Make a prediction as if we didn't know this value
    pred_alpha = client.predict(col, given=df.loc[row_ix].drop(col).to_dict())
    ymin, ymax = ax.get_ylim()
    pred_line = ax.vlines(
        x=pred_alpha[0], ymin=ymin, ymax=ymax, color="orange", label="Predicted"
    )
    true_line = ax.vlines(
        x=df.loc[row_ix, col], ymin=ymin, ymax=ymax, color="green", label="True"
    )

    ax.set_title(
        f"Sample {row_ix}, {col} prediction: {round(pred_alpha[0], 2)}, "
        f"Uncertainty: {round(pred_alpha[1], 3)}",
        fontsize=15
    )
    ax.legend(handles=[pred_line, true_line])

    return ax
```

Now, choosing a random sample from our dataset (row 6), let's ask it what
it would predict for a couple of its feature attributes, and what distribution
each internal state would have chosen from.

In these examples, we will see

* **Low** uncertainty for the case of predicting `fSize`
* **Medium** uncertainty for predicting `fAlpha`
* **High** uncertainty for predicting `fLength`

```python
uncertainty_plot(
    client=c,
    df=df,
    col="fSize",
    row_ix="6",
    value_list=np.arange(2, 4, 0.01),
)

uncertainty_plot(
    client=c,
    df=df,
    col="fAlpha",
    row_ix="6",
    value_list=np.arange(0, 25, 0.1),
)

uncertainty_plot(
    client=c,
    df=df,
    col="fLength",
    row_ix="6",
    value_list=np.arange(20, 90, 0.1),
)
```

![fSize Uncertainty](/img/gamma/fsize-uncertainty.png)
![fAlpha Uncertainty](/img/gamma/falpha-uncertainty.png)
![fLength Uncertainty](/img/gamma/flength-uncertainty.png)

It's also worth noting that when asking Braid for simulated samples, these
are the types of distributions that are calculated and used to draw from.

# Conclusion

There is a limited paradigm in which the idea of AI and machine learning is
currently entrenched. There exists entire toolboxes of "unit-taskers" that
can successfully do one thing pretty well. I hope you have found in this post
evidence that a new paradigm is possible with the Braid engine. One that can
model an entire data set or domain and do many, many things well.

* Prediction (of **all** features)
* Work with missing data — and is able to intelligently impute missing data
* Provide a feature importance w.r.t. any other feature or group of features
* Provide uncertainty of predictions
* Simulate new samples
* Detect anomalous whole samples
* Detect anomalous individual values, given the context of the rest of a sample

And all of this is done not as some Frankenstein Suite of unrelated products and
models, but as the result of exploiting all of the benefits that come from one
platform understanding the whole feature space.
