+++
title = "Explainable AI is not enough."
date = 2019-09-05
template = "post.html"
draft = true

[taxonomies]
categories = ["ai", "ethics"]

[extra]
author = "Baxter Eaves"
subheading = "Explaining black box AI won't prevent problems; it will make them worse."
image = "abandoned-accident-aeroplane-2787456.jpg"
theme = "dark-transparent"
+++

The black box nature of modern AI makes it impossible to use in many domains. For example, in defense, when someone makes a decision, they sign their name to a piece of letterhead and they become legally responsible for the outcomes of that decision. Thousands of lives and the fate of nations potentially hang in the balance, so obviously they're not going to just trust the word of what is effectively a magic 8-ball. They need to know where the decision comes from, what the machine knows, when it fails, when it succeeds; when it can be trusted. To this end DARPA launched the [Explainable AI (XAI) project](), which aims to fund development of such AI technologies. In this post, I'll discuss what XAI is, and how *explanation* fails to address concerns of auditability and safety.

# What is explainable AI

According to (former) XAI program manager David Gunning [(Link to PDF source)](https://www.darpa.mil/attachments/XAIProgramUpdate.pdf):

> The current generation of AI systems offer tremendous benefits, but their effectiveness will be limited by the machine's inability to explain its decisions and actions to users. 

If we're using AI in transportation, finance, security, medicine, or the military, there are a few key questions an AI needs toi answer. According to the program summary:

- Why did you do that?
- Why not something else?
- When do you succeed?
- When do you fail?
- When can I trust you?
- How to I correct and error?

For AI to be useful in theses high-risk, high-impact domains it must be completely transparent and intuitive to its human users. However, the bulk of explainable AI research (both separate and as a part of the XAI program) focuses singly on explanation and unfortunately focusing on the *explanation* part of explainable AI won't solve AI's problem. It will likely make them worse.

# Problems with explanation for black box AI

WRITEME

## Explanations are excuses

The most glaring issue with explanation in AI is that explanations tell you why something happened. Past tense. Something has to go wrong for us to fix it. We're not preventing problems, we're explaining them. Being able to say that your autonomous car struck a pedestrian because it marked her as a fase positive is no use to anyone.

## Explanations don't tell use what the machine knows

An explanation gives us knowledge about a specific decision or prediction. It does not tell us what the machine knows. It does not tell us about the knowledge inside that machine that shaped that decision or prediction.

## Explanation makes inappropriate trust worst

People have a tendency to anthropomorphize things, and to find patters where they don't exist. We also tend to trust people. Trusting people is important for learning. If I distrust everything you say, I cannot learn from you; if I trust you implicitly, I can accept everything you say without question, which allows me to learn quickly. If we look at a visualization of an AI explanation, say a heat map explaining the important parts of an image to its classification

# Moving toward the spirit of XAI

To use AI in high-risk high impact domains, we need AI that are completely transparent and intuitive to the humans that are going to be using them. This is the spirit of XAI. The unfortunate use of the word explanation has permits researchers to continue to use dangerous and inappropriate AI models as long as they can develop additional wrapper models to generate plausible explanations for decisions or predictions. But some AI researchers have taken different approaches, focusing instead on *interpretable* models whose workings can be intuitively understood by human users, or by focusing on embedding machine knowledge directly into human users through *teaching*.

Interpretable models are trivially explainable. The difficulty is that making models that are interpretable, general, and powerful is really, really hard. Then there's making them fast...

Teaching is a promising approach. If we can transfer machine knowledge into a human, then we no longer need to as an AI questions about its behaviour. Those questions become introspection. I can decide how I feel about making a decision based on what the machine has taught me. If I feel icky, I can step back and start figuring out why. The issue with teaching is that it is hard. Computationally, it involves recursive reasoning between teacher and learner, which is hard to make work for complex knowledge.

But of course, these are temporary problems. The future is bright for explainable AI, and I think we're on the cusp of a major paradigm change. AI has been going down the wrong path for a long time; people are butting up hard against its limits and are eager to fix them.

# Key points

- Explainable AI is about enabling AI in high-risk sectors by developing AI that are completely transparent and intuitive to decision makers.
- The word "explanation" has lead researchers to develop secondary models to explain predictions and decisions made by uninterpretable primary models.
- There are some problems with explanation:
    + Explanations don't prevent problems, they offer excuses as to why they happened.
    + Explanation does not tell us what the machine has learned.
    + Explanations make inappropriate trust worse by increasing trust in brittle, unpredictable models.
- Some researchers are focusing on general interpretable models, which are trivially explainable
- Some researchers are focusing on teaching, which embeds machine knowledge into a human mind, which turn questions we'd ask a machine into questions we'd ask ourselves
