+++
title = "Why you're not getting ROI from of data science"
date = 2019-10-20
template = "post.html"
draft = false

[taxonomies]
categories = ["data science", "business", "ai"]

[extra]
author = "Baxter Eaves"
subheading = "It's not you, it's machine learning"
image = "black-rabbit-01.jpg"
theme = "light-transparent"
+++

Story.

# Reasons ROI is hard in data science

The main reason that its hard to get ROI from data science is because machine learning is dumb and you are smart, but you think the opposite is true. 

## ML models can't cope with the (sad) state of your data

Before we even start doing data science, we're often out of luck.

Most machine learning models can't handle our data. Want to do deep learning? Better have millions of data or we can't learn. Better not have missing data because you're going to have to fill them in or throw them out. Better not have categorical data because you'll need more data to support the extra encoding.
Do your data have unexplained trends in them? Were they collected from multiple sources, or before machine learning was a concern? We're probably going to learn to predict using that random noise.

## ML models can't do what you want with your data

Do you

## Asking questions takes too long

Most machine learning models are designed to answer one question: predict Y given X. What X and Y are depend on the model and what you're trying to accomplish. Y might be a numerical value like height, a categorical value like *good* or *bad*, or might be the next few words in a sentence. The thing is that you have to know what X and Y are before you start even doing machine learning, which means you have to know which question you want to ask. The truth is that this you might not be able to ask the exact question you want because the model doesn't support that data. Maybe the kinds of data X and Y are incompatible with the model that you believe will best answer your data. You have to go tweak the data, which could bias the results; throw the incompatible data out, which is potentially wasteful; or choose a new model that might not answer the question as well. Doing any of these things has changed the question.

The rest of the data science process is a cycle of training/fitting a model (the process by which the ML model is said to learn), validating, thinking you're not doing as well as you could be; then tweaking the model or choosing a new model entirely.

By the time you have something that is as good as you think you're going to do, what you did looks nothing like what you wanted to do when you started, because you've spend months manually trying to figure out which question is the one you can answer well. So you've answered one question well, reported the results, and now you're asked to answer another question. See you in a few months.

## Not knowing the which questions (if any) you can answer

A lot of the fitting, validating, tweaking cycle is due to people -- both decision makers and analysts -- refusing to believe they can't do something. The hype has you believing that machines are smart and you are dumb. That if you can't get actionable insights from your data, you are personally at fault. The truth is, sometimes you can't answer the question you want to. Sometimes you can't answer any question at all. It might be because you're data are bad. Perhaps you don't have the right data to tease apart complex interaction. But it can also be that the thing you were hoping to find isn't a thing.

Data scientist and analysts like to say something that fails to perform "doesn't work". Who is to say that poorly isn't the best that you can do given the underlying (unknown) process that generated your data?


## Throwing science against the wall to see what sticks

What happens when something doesn't work, is that you find something that does. You tweak the data. You tweak the model. You find a new model; a more complex model. Maybe you stack and combine models. Maybe you extrude your data through a model so they can fit into another. Hell, if deep learning can beat the pros at starcraft, it can predict how long the tassels on your corn plant are going to be, right?

So we try everything. We spend a lot of time. And we force it to work.

## Not knowing what is real

And by the end of it we don't know whether what we found is real, or whether it's what we wanted to see. Even if we weren't deliberately over-fitting our model, each thing we try is a coin flip. If you flip a coin enough times, it's going to come up heads eventually. In statistics, we correct for this. In machine learning, nobody asks.

## You are wasting your data scientists

Data scientists are expensive so use them for things that can't be done by someone else more cheaply. If data scientists are Collecting or cleaning data, or are doing discovery with packaged models, you are wasting your money.

You

Unless they are subject area experts, data scientists should not be in charge of choosing which questions to ask, which means they should not be doing data science discovery as outlined above. There are a number of companies offering tools that automate the kind of **bad** data science outlined above.

**Data scientists should have one task: taking what was learned in discovery, sitting with experts, and building an optimized custom model to solve a specific business problem.** Every assumption in the model should be documented and signed off on by a subject are expert. And the thing they write should be production ready. It shouldn't be thrown over the fence to engineering. But this isn't something any data scientist can do, it's something only a good data scientist can do.

## You have bad data scientists

Elephant in the room.

A bad data scientist is a catalogue of models. They just go through all of the algorithms in whichever software package they like, and declare a winner based on raw performance.

Good data science can come from a number of places. I think it's most common for great data scientist to be people that are experts in statistics and engineering, but some of the most impactful work comes from people who are excited about machine learning but also know their subject area front and back. As I mentioned in my **FIXME: post** I very much believe that if you want the best bang for your buck, that you'll train domain experts in data science rather than train data scientists in a complex domain.


# Wrapping up

# Key Points
- A ML model can't handle your complex, sparse, partial, unstructured, and noisy data
- A ML model doesn't let you ask multiple questions quickly
- A ML model doesn't tell you what questions you can answer
- A ML model doesn't tell you what is real
