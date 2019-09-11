+++
title = "How to design an efficient data collection plan for science and AI"
date = 2019-09-05
template = "post.html"
draft = false

[taxonomies]
categories = ["ai", "science"]

[extra]
author = "Baxter Eaves"
subheading = "To ensure that our AI predict accurately, we must provide it with the right data."
image = "greenhouse-2.jpg"
theme = "light-transparent"
+++

Our businesses run on selling the good stuff, so it seems wasteful to spend resources generating stuff that we don't know will be good or that we know will be bad. But if we're using this stuff we generate in our AI/ML training data, not doing so means we're not really learning, which means we're losing our ability to predict accurately outside a very narrow realm of knowledge, which means we cannot innovate. Fortunately the cost of generating extra data can be mitigated to a great extent by choosing data in a targeted way. In this post we'll discuss ways to choose data to improve learning.

# Why we need to keep our AI in mind when we design experiments

Often in science we create our own training data for our machine learning and AI models. We run an experiment that creates data and we feed those data to machines. It is also often the case that the experiments we choose to run are based on what we think the outcome will be. For example, in plant breeding you want to make the cross that will have the best yield -- e.g. soy with the most bean. Breeders make the cross, then test and measure the cross; and the data from that cross end up as a part of the training set for the prediction model that tells us which crosses will be best. What's the problem? We're biasing our data. We're learning only about things that we're certain are good, and in doing so we are losing our ability to tell what is bad, or even what weird things have potential to be good. Soon everything will look good, because everything the model has seen is good. But everything won't be good in the field. Progress will halt, and it will take a massive amount of time and money to generate, or buy, the data needed to predict accurately.

# Do not choose data that minimize uncertainty

One way to choose which experiments to run is to choose those that improve our ability to confidently make decisions in the future. The problem with this approach is that the easiest way to have high certainty is to be ignorant. If I learn about a set of things that are very similar I will have a very high certainty about the predictions I make with respect to similar data. Then, to ensure high certainty, I make sure to progress things I already have high certainty about, which are things I already know about, which are things I've seen a lot of. See how we're building ignorance into our system? The way out here is to build some sort of diversity heuristics into our selection model. But heuristics themselves are biasing. You'll have to come up with a definition of diversity that is both appropriate to the domain and relevant to learning. Not an easy task. And who's to say that those definitions won't change? In any case, there is no guarantee that whatever metric you come up with wont cause your model to fall into some catastrophic edge case that rewards bias or ignorance and ruins your dataset.

# Choose data that maximize learning

Alternatively: embrace uncertainty. Some things are uncertain. It might be that your data are noisy, it might be that the world is noisy. Sometimes it's just hard to make confident decisions. You need to accept that. You should know when you are unconfident, and be glad that you know. It's a heck of a lot better to say "I don't know" than to confidently make a bad decision.

Let's instead focus on learning. This strategy is incredible simple: choose to run experiments that you are uncertain about. If you're using the right model, you can quantify the amount of information in an experiment as [bits](https://en.wikipedia.org/wiki/Bit). More bits mean more learning. Now you can put a number to the value of your designated learning set. If the cost of your experiments is variable, you can even optimize bits of information per dollar.

The way that we might do this in practice to to set aside a certain proportion of our total data collection efforts to a learning set. Say that we allocate 90% of our data to the main set and 10% to our learning set. We first choose the data in the main set based on there predicted performance. Next, we choose out learning data by measuring their information gain when combined with the main set. If we have time, we can even optimize the size of the learning set by figuring out at what percent of total data that the learning data provides diminishing returns. For example, a 5% learning set may give us 1000 bits, a 10% set may give us 1500 bits, and a 20 % set may give us 1750 bits. In a case like this, we might just want to limit our learning set to 10% of total.

# Wrapping up

If we want to continue developing the best product we have to learn about he bad products and explore the risky ones. In this post we discussed two ways to design experiments for and with AI. The strategy of choosing the data that minimize uncertainty in our decisions is likely to promote ignorance; avoiding entrenchment under this strategy is difficult because it requires lots of knowledge and careful tuning. The strategy of generating experiments that you're uncertain about maximizes learning. Better still is quantifying the number of bits of information the experiment will produce and optimizing bits for unit of cost.

# Key points

- Accurate prediction not only requires quality data, but requires the right set of data. We must collect data that represent the process we want to model.
- Do not collect data features that are redundant or do not improve prediction.
- Do not base data collection plans on minimizing uncertainty. Without great care and consideration this is a sure path to biased, uninformative data.
- Collect data on events you are uncertain about.
- Quantify the amount of information in your learning set with *bits* and optimize for bit per unit of cost.
