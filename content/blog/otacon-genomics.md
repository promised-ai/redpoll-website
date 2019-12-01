+++
title = "Why we're using molecular breeding to test the general purpose AI we're building for DARAPA"
date = 2019-12-15
template = "post.html"
draft = false

[taxonomies]
categories = ["news", "ai"]

[extra]
author = "Baxter Eaves"
subheading = "DARPA's SAIL-ON prgram focuses on novelty and novelty is the mechanism of evolution"
image = "red-flower.jpg"
theme = "light-transparent"
+++

"Defense AI" elicits images of *War Games*: satellite imagery overlaid with markers indicating the positions and movements of aircraft, troops, and supplies. Of the real-time game of war. So why is it that we're using plant breeding as a test bed to develop the AI we're building for the Defense Department?

# The SAIL-ON Program

We're developing AI for DARPA under the SAIL-ON program. The stated objective of the SAIL-ON program is to make learning machines that are more robust to different types of *novelty* in the world. There are different types of novelty.

There are single odd examples. Say an image classifier trained to classify images of fruits recieves an image of a toad. How will the machine react? A machine that knows only fruits, may pick up on the green bumpy skin and may, with high certainty, classify the toad as an avocado. What we would like the machine to do is to recognize the toad's weirdness and react to it. Perhaps by discarding the example, creating a new class of item for that example, or by asking a human for help. Recognizing novel examples is especially important for preventing attacks where an attacker can make changes to an image that are imperceptable to a human, but that change the classifier results dramatically (such as attacks that cause *stop* signs to be classifier as *speed limit 45* signs [FIXME: link]).

There is novelty stemming from changes in the way the world is represented or in the dynamics of the world. When a person learns to play go, they typically learn on a small 9-by-9 board, and then eventually move up to a standard 19-by-19 board. A machine would have to learn 9-by-9 and 19-by-19 go a separate games; it could not transfer its knowledge from one board to the other. And what if you decide to keep score differently in the middle of the game? Again, this is something a human can handle with ease, but a machine must be retrained to do. 

## Sources of novelty

The first draft of SAIL-ON's *novelty heirarchy* outlines several sources of novelty that I have collapsed:

- Class - Previouslly unseen class of object
- Features - Change in how data features are specified or addition/removal of features
- Internal Dynamics - Internal change in the system governing how the data are generated.
- External Dynamics - Changes in features that are not accounted for by the data that affect the collected features (e.g. environment)
- Agent dynamics - Changes in objectives of in the affect of an agent's actions

# Molecular breeding

Molecular breeding is basically using genetics data to inform breeding decisions. It is commonly used in plant and livestock breeding. The genetics are used for a couple of purposes: to predict the performance of an individual without having to grow it, and to ensure genetic diversity. If we have enough data relating the genetic sequences of plants to their observable traits -- like grain yield and disease resistance -- we can predict a trait given a genetic sequence. But selecting only the highest-performing individuals will likely reduce our genetic diversity. This is bad because if a new disease comes along that our individuals are not resistant to, we could lose everything. We need novel genetic material for resistance.

Of course neither prediction nor diversity are as easy as that. Genetics are extremely complex and there you'd need an incredible amount of data to predict account for that complexity. And prediction is made more complicated by the environment which influences the development of organisms through epigenetic and external factors. 

Genetic diversity
