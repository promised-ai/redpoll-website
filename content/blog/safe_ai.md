+++
title = "What it takes to build Safe AI"
date = 2019-07-31
template = "post.html"

[extra]
author = "Baxter Eaves"
subheading = "To be safe, AI must be human"
image = "black-and-white-clouds-cold-2086620.jpg"
theme = "dark-opaque"
+++

# Notes

- General AI history
     McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. The bulletin of mathematical biophysics, 5(4), 115-133.
    +   + Was the original ANN paper
    + Dartmouth Conference 1956 psychologists, engineers, mathematicians
    + Overoptimism in golden years of 1956 to 1974
    + First 'Winter' in 74-80 (dark ages of connectionism)
    + A new boom in 80-87 (expert systems, hopfield nets)
    + Second winter 87-93 (brittle, expensive, embodied reasoning)
    + 93-2011 AI is doing stuff, but not getting credit
    + 2011 to now: Boom mostly because deep learning
    + Winter is coming
    + Why: You can't us AI for anything important



# Body

We like to talk about Safe AI a lot here, but what does that mean? What makes an AI safe or unsafe? In this post, we'll explore these ideas at a high level.

In general, predictable things are safe and unpredictable things are dangerous. If you can account for all the parameters under which a thing fails or becomes a hazard, it can be perfectly safe. A tire that spontaneously bursts is dangerous, but if the tire is known only to burst at highway speeds, it can be made safe by not using it on the highway. The same is true for AI. If we know exactly when and how an AI will fail, we can know when and how to use it. Unfortunately, the most ubiquitous AI models are black boxes that by their nature are unpredictable.

## Explainable AI

Making these models predictable is no easy task, and so researchers have instead ventured to make them *explainable*. This means that machinery is added to the AI that emits explaination -- why the AI made a prediction or took an action. This explainability machinery can take the form of a wrapper based on heuristics that analyze the structure of the AI, or even an additional AI that attemps to pack its knowledge into a more transparent model that can be queried more straightforwardly.

There are two problems here: adding explainable wrappers to unpredictable models, that can fail unexpectedly and catasrophically, adds a false sense of trust. The model is still unpredictable. The repacking approach risks repacking bad knowledge, and it suffers the additional overhead of translation. Translating one model to another more often than not results in information loss, so the final model may not perform as well as the original. Now we have a model that can fail unexpectedly and catasrophically, and is also less performant.

## Where are the human results


## Humanistic > Explainable

