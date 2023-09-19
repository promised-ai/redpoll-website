+++
title = "Introducing Lace: Bayesian tabular data analysis in Rust and Python"
date = 2023-09-21
template = "post.html"
draft = false

[taxonomies]
categories = ["data science", "open source"]

[extra]
author = "Baxter Eaves"
subheading = "A machine learning tool optimized for human learning"
image = "fireworks-1.jpeg"
image_attr = "'Fireworks on Dark Sky' Ryan Klaus"
plotly = true
+++
<style>
.js-plotly-plot {
    aspect-ratio: 1;
    max-height: 650px;
}
</style>

Recently we released the source code of one of our core internal tools, Lace. Lace is a Bayesian tabular data analysis tool designed to make learning from your data as easy as possible. Learn more at [lace.dev](https://lace.dev).

# A tour of lace

Lace is a tool for analyzing tabular data &mdash; think pandas `DataFrame`s. It is both *generative* and *discriminative* in that it has a generative model from which you can generate data and predict unobserved values. It natively handles continuous, categorical, and missing data, and can even do missing-not-at-random inference when the absence/presence of data has potential significance.

Our goal with Lace is to optimize for human learning by optimizing for transparency and speed of asking questions.

Whereas typical machine learning models are designed to learn a function linking inputs to outputs, Lace learns a joint probability distribution over the entire data set. Once you have the joint distribution you can trivially create conditional distributions through which you can ask and discover questions.

For most use cases, using Lace is as simple as dropping a pandas or polars dataframe into the `Engine` constructor and runing `update`.

```Python
import pandas as pd
import lace

df = pd.read_csv("my-data.csv")

engine = lace.Engine.from_df(df)
engine.update(1000)
```

## Understanding statistical structure

But before we ask any questions, we like to know which questions we can answer. So, we'll ask which features are statistically dependent, i.e., which features are predictive of each other, using `depprob`.

```python
# lace comes with an Animals and Satellites example dataset
from lace.examples import Satellites

sats = Satellites()
sats.clustermap("depprob", zmin=0, zmax=1)
```
{{ 
    jsplot(
        path="content/blog/introducing-lace/depprob.html",
        img="",
        caption="Dependence probability matrix. Each cell shows the probability that a dependence path exists between two features."
    ) 
}} 

Above, each cell tells us the probabilty that two variables are statistically dependent (though that dependence might flow through one or more intermediate variables).


## Predicition and likelihood evaluation

We can of course do prediction (regression or classification) using the `predict` command.

```python
# Marginal distribution of orbital period
sats.predict("Period_minutes")
# (100.59185703181058, 1.4439934663361522)

# Add conditions (add as many as you like)
sats.predict("Period_minutes", given={"Class_of_Orbit": "GEO"})
# (1436.0404065183673, 0.8641390940629012)

# Condition on missing values
sats.predict("Class_of_Orbit", given={"longitude_radians_of_geo": None})
# ('LEO', 0.002252910143782927)
```

Note that calls to `predict` return two values: the prediction and a second number describing uncertainty (Jensen-Shannon divergence among the posterior samples' predictive distributions).

If you'd like to view the entire predictive distribution rather than just the most likely value (prediction), you can ask about the likelihood of values.

```python
import numpy as np
import pandas as pd

xs = pd.Series(np.linspace(0, 1500, 20), name="Period_minutes")
sats.logp(xs)
```

|    |      logp |
|---:|----------:|
|  0 | -10.5922  |
|  1 |  -6.48785 |
|  2 | -10.8964  |
|  3 | -10.8551  |
|  4 | -10.782   |
|  &vellip; | &vellip;|
| 15 | -10.8064  |
| 16 | -10.8447  |
| 17 | -10.5111  |
| 18 |  -8.90361 |
| 19 |  -9.86572 |

Visualizing distributions for each posterior sample (often referred to as "states") allows us to get a nice view of uncertainty.

```python
from lace.plot import prediction_uncertainty

prediction_uncertainty(
    sats, 
    "Period_minutes",
    given={"Class_of_Orbit": "GEO"},
    xs=pd.Series(np.linspace(1400, 1480, 500), name="Period_minutes")
)
```

{{ 
    jsplot(
        path="content/blog/introducing-lace/uncertainty.html",
        img="",
        caption="A visualization of uncertainty when predicting Period_minutes given a geosynchronous orbit class. The red line is the most likely value, the black line is the likelihood of Period_minutes over a range of values, and the gray lines represent the likleihoods emitted by a number of posterior samples."
    ) 
}} 

Above, the red line is the prediction (the most likely value), the black line is the probability distribution for this prediction, and the gray lines are the distribution for each posterior sample, which tells you how certain the model is that it has captured the distribution (learn more about uncertainty quantification [here](/blog/ml-uncertainty)).


## Simulation

We can simulate values

```python
sats.simulate(["Users"], n=5)
```

|    | Users      |
|---:|:-----------|
|  0 | Commercial |
|  1 | Military   |
|  2 | Commercial |
|  3 | Military   |
|  4 | Government |

Just like with `logp` and `predict`, we can add conditions for our simulations.

```python
sats.simulate(
    ["Users", "Purpose"],
    given={
        "Class_of_Orbit": "LEO",
        "Launch_Site": "Taiyuan Launch Center"
    },
    n=5
)
```

|    | Users      | Purpose                |
|---:|:-----------|:-----------------------|
|  0 | Government | Technology Development |
|  1 | Government | Earth Science          |
|  2 | Civil      | Communications         |
|  3 | Government | Technology Development |
|  4 | Commercial | Communications         |


Heck, we can re-simulate the entire dataset. Lace is generally very good at generating tabular synthetic data, outperforming GAN- and Tranformer-based approaches (manuscript under review).

```python
sats.simulate(sats.columns, n=sats.shape[0])
```

## Row/Record similarity

We can also ask which records (rows) are similar in terms of model space. This frees us from having to come up with a distance metric that works well for mixed data types and missing data. It also provides more nuanced information that just looking at the values. To make this a bit more inuitive, we'll switch to an animals example, since people generally have a better sense of what makes animals similar than they do what makes satellites similar.

```python
from lace.examples import Animals

animals = Animals()
animals.clustermap("rowsim", zmin=0, zmax=1)
```

{{ 
    jsplot(
        path="content/blog/introducing-lace/rowsim.html",
        img="",
        caption="Row similarity of animals. Higher row similarity means the animals are closer in model space, meaning their features are modeled similarly."
    ) 
}} 

This essentailly generates a data-drive taxonomy of animals for us. More similar animals will be modeled more similarly. If two animals have a row similarity of 1 it means their features are modeled identically.

We can ask for similarity given a specific context. Say that we only cared about similarity with respect to whether an animals `swims`.

```python
animals.clustermap("rowsim", zmin=0, zmax=1, fn_kwargs={"wrt": ["swims"]})
```

<div>
{{ 
    jsplot(
        path="content/blog/introducing-lace/rowsim-wrt.html",
        img="", 
        caption="Row similarity with respect to how the 'swimming' feature is modeled."
    ) 
}} 
</div>

Notice that there are two main clusters of animals, those that swims and those that do not. If we were just looking at the data similarity the values would all be either 0 or 1 because the `swims` feature is binary, but here we get more nuanced information. For example, within the animals that swim, there are two similarity groups. These are groups of animals that swims, but we predict that they swim for different reasons.

# Learn more

We've put a lot of love into lace, and there's a lot more that you can do with it than we've gone over here. To learn more visit [lace.dev](https://lace.dev) or checkout the [github repository](https://github.com/promised-ai/lace).
