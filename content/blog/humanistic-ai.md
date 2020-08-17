+++
title = "On Humanistic Artificial Intelligence"
date = 2020-08-18
template = "post.html"
draft = false

[taxonomies]
categories = ["ai"]

[extra]
author = "Baxter Eaves"
subheading = "Choosing cognition — not the brain — as the basis of Artificial Intelligence design"
image = "lighthouse-02.jpg"
theme = "light-transparent"
front_page = false
+++

**To make AI that can partner with people, we must make AI that is humanistic
in its functioning, not anatomical in its construction.**

AI is failing to make impact on our most important problems. Since its
inception, advances in AI have been primarily driven by advances in *Neural*
architecture: architecture intended to mimic the functioning of networks of
neurons. Architecture intended to mimic brains. An in-depth philosophical
discussion of why this architecture fails &mdash; and will continue to fail
&mdash; deserves its own post, but it is enough to say that AI has focused on
**replacing** human decision-makers.  And in pursuing the goal of automating,
rather than supporting, human decisions, AI has made itself rigid, fragile, and
incompatible with people.  When we say "people change the world" it is because
people drive solutions to humanity's most important problems. Thus if AI is
incompatible with people, AI is incompatible with humanity's most important
problems.

To make an AI that is compatible with people, we must ignore the brain as a
base. People are much more than their brains. We must build systems that
directly reproduce human cognitive capabilities. Cognition is thinking and
learning. When people decide how to act, they use their cognition. A person is
not aware of the mechanics of their brain &mdash; which neurons are firing
across which synapses when, or how much of which neurotransmitters are being
released and reuptaken (reuptook?) &mdash; they are aware (to some extent) of
their cognition. A machine that does cognition accurately is a machine that
behaves humanistically, and is therefore fundamentally human-compatible.

We need machines that provably learn like people and store their knowledge like
people.

# What does Humanistic AI get us?

Learning the way a person does is good and well, but, apart from being easier
to understand, what does it get us? Humanistic AI enables a fundamentally
different way of interacting with AI. Some significant advantages are:

- We no longer must know the right question to ask before we start modeling
- We no longer must retrain when we get new data or edit existing data
- We can identify gaps in our own knowledge
- We get information about how to close our knowledge gaps.

Humanistic AI systems are also resistant to attacks and unexpected behavior
induced by strange data. And because Humanistic AI systems are epistemically
aware (they know just how much they don't know), they are much less likely to
have inferences fail silently in production.

## No more inputs and outputs

The standard machine learning model is built around the idea of learning a
function that relates a set of input data to a set of outputs. For example, a
bank may want to relate customers' demographic information and credit history
(inputs) to whether they will make their mortgage payments on time (outputs).
It seems simple, but there is a lot we need to be able to do this. We need to
know which question is valuable to answer, we need to know that we have
sufficient data that support the question, and we need to know that the
question can indeed be answered with those data. Learning these three things is
arguably the most time-consuming process of implementing any AI/ML solution.

In contrast, people do not need to have a question in mind to learn. People
just observe data in the world and build information from it. They learn which
data are related and learn how to predict things given other things &mdash;
both forward and backward. There are no inputs or outputs. People learn one big
model of the world. And as we learn, we can pose questions to the body of
knowledge we have acquired. "Is the stock market likely to go up today?" "Is
this safe to eat?"

Humanistic AI has no notion of inputs and outputs, nor must it be built around
a single question. It learns the process that generates a set of data, and then
it allows the user to ask any number of questions about that process with no
additional effort. More questions, more answers, and thus more knowledge more
quickly.

## Reduce Data Munging & Eliminate Retraining

When a machine learning model receives new data it must retrain &mdash; even if
given a single new data point. All prior learning must be thrown away and
re-done from scratch. This is incredibly wasteful, especially considering that
modern Neural Architecture is extremely complex, containing millions of
parameters, and can take weeks or months to train.

New data come in all the time. New users shop at a store. New patients are
admitted to a hospital. The sun shines, or it does not. The Earth keeps on
spinning. We do not want to wait days to take advantage of the information in
the data we have streaming in, nor do we want to spend thousands of dollars
retraining these models over and over.

People learn from streams. We observe data and learn in real-time. We do not
store large batches of raw data then shut down for days at a time to integrate
new learning.

Humanistic AI learns from streams. It is aware of its knowledge, so it knows
which specific knowledge needs updating in the face of new data. This
*Epistemic Awareness* also allows it to handle backfilling without retraining.
Users may fill in missing entries or correct data entry errors without
retraining, ensuring that information and inferences are always up-to-date.

## Know What You Do Not Know

Neural architectures are tragically fragile. They are easily fooled, easily
attacked, and do a bad job of reasonably quantifying their uncertainty &mdash;
appropriately admitting "I don't know". If you train a Deep Network to classify
images of fruits then you give it an image of a frog it will gladly tell you "I
am very certain that this is an avocado." 

We would much rather it say "This is something entirely new. I'm not sure what
this is". People do that; so should AI.

Because humanistic AI is introspective and learns from streams, it can assess
its knowledge in real-time. It will tell you that it does not know, it will
tell you why it does not know, and it can advise on how to generate data that
will help it to figure things out. Humanistic AI can also monitor the state of
its knowledge over time and tell you when things are changing.

Imagine you have implemented a traditional AI/ML system to recommend whether a
patient with respiratory trouble should be either placed on ventilation or
given oxygen. One day, a group of doctors marches into your office telling you
that the system is consistently wrong and that people are getting ill. You shut
things down to diagnose what went wrong. Maybe you figure it out; maybe you
don't. But the system does not help you.

A humanistic AI system would warn you that its prediction certainty was
decreasing, and it furthermore would tell you the cause. Maybe in this case a
new variable is influencing its prediction. Humanistic AI could show you that
sometime in the past day or two the wing of the hospital to which the patient
was admitted began affecting its predictions. You examine those data and see
that patients in the South wing have been experiencing pneumonia at a higher
rate. These insights lead to a course of actions that reveals faulty sanitation
equipment in the South wing. The equipment is repaired and patients placed on
ventilators stop contracting pneumonia.

## Know What You Need To Do To Learn

Knowing that we do not know is great, but learning is about fixing our
ignorance. People can learn *actively*. One of the benefits of knowing how the
world works is that one can predict (to a degree) what data the world will
produce under what circumstances. If one is aware of one's knowledge, one is
aware of the effect data have on one's knowledge; and if one knows what data a
process will produce under what interventions, one can intervene intentionally
to produce data that produce specific learning outcomes. One can do science.
One can learn actively.

An AI system may not be able to conduct its own experiments (acting on the
world requires a physical body), but it can suggest experiments to its human
colleagues. It can also weigh the amount of learning expected from an
experiment with the cost of running that experiment, ensuring that the data
collection plan maximizes information gain per dollar spent.

# Limitations

Humanistic AI has a number of advantages over traditional AI &mdash; it is
safer and easier to use &mdash; but it has limits. Humanistic AI is new and has
not had the benefit of the vast body of effort that Neural AI has. At the time
of writing, the code base of [TesnorFlow](https://www.tensorflow.org/), the
most popular Deep Learning backend, has 2767 contributors. Neural Architecture
has been the most researched subfield of AI, and has its roots in work done in
the 1940's. The most publicized AI at the largest companies: all Neural. As a
result, Neural architecture is fast and scaleable. On the other hand, speed
gained by Humanist AI comes primarily from its workflow, which reduces waste
and repetition. Humanistic AI does not yet scale to petascale data. It cannot
yet take advantage of supercomputer architecture. But is was not long ago that
the same limitations held for Neural Architecture.

As we butt up against the limits of today's AI, the need for a new kind of AI
paradigm is becoming apparent. There are problems we *must* solve, and we need
better tools to help the people solving them because those people *cannot* be
replaced. We believe Humanistic AI is that tool.


# Key Points
- Modern AI architecture is fundamentally incompatible with people in part
  because it was developed to make decisions instead of support them.
- People cannot be replaced in high-risk, high-impact domains, so AI must
  partner with people, and to do so it must be compatible with people.
- The architecture of Humanistic AI is one that mimics cognition. This is in
  direct contrast to today's most ubiquitous AI architectures whose foundations
  lie in mimicking the anatomical structure of the brain.
- Some of the advantages of Humanistic AI
    + Less data munging
    + Less required of the user
    + Easy to ask questions
    + Easy to understand the answers to those questions and the bases for those answers
    + Learns from streams of data
    + Aware of what it knows and does not know
    + Can help design efficient data collection plans
- Humanistic AI does not currently scale to the level of data that neural
  architecture does, but offers speed benefits in the form of reduced waste and
  repetition
