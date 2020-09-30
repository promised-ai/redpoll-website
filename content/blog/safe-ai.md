+++
title = "Components of safe Artificial Intelligence"
date = 2020-09-25
template = "post.html"
draft = false

[taxonomies]
categories = ["ai"]

[extra]
author = "Baxter Eaves"
subheading = "What artificial intelligence needs — other than people — to be safe"
image = "burning.jpg"
theme = "light-transparent"
front_page = false
+++

Modern AI, on its own, is dangerous. It is brittle and difficult to understand, which makes it an unpredictable element. So then how can we use AI in safety critical applications? Today, advanced AI systems like those in self-driving cars have even more advanced systems mitigating the damage they can do. Those systems are called *people*. The person behind the wheel of a self-driving car is responsible for harm done by that car, the physician is responsible for the outcome of any treatment done; the flag officer is responsible for the geopolitical ramifications of a military movement. This is why the AI industry has been OK with using black-box technology in these applications: the responsibility has been placed upon the user &mdash; who is likely the person that least understands the technology. Ethical discussion aside, this post will discuss what AI needs &mdash; other than reliance on people &mdash; to be safe.

# Interpretability; not explainability

In the past year or two "Explainable AI" has become a buzzword. It is a kind of AI that attempts to explain where its decisions came from. Any kind of AI can be explainable as long as you have a way to explain it. On the other hand, *interpretable* AI must be self-explanatory. Its knowledge must be transparent to the user without any extra machinery. Find a more in-depth discussion of the limits of explainability [here](@/blog/explainable-problems.md), but relevant to this discussion: interpretability offers safety but explainabilty does not.

## Explainabilty

Explainability does not guarantee safety for a number of reasons that boil down to the idea that explaining that something is unsafe does not make it safe. If I know a software quirk in a car's lane assist can cause random swerving at high speeds, I will choose not to ride in that car.

1. Explanations are not necessarily informative. "Because these pixels made me think she was a car" tells you why, but it doesn't really. Adding information, "because these pixels activated a 'car wheel' patch, which made me think she was a car" is equally unsatisfactory.

2. Explanations of unexplainable models need explanation. Explanations are often ad-hoc methodologies built on top of black box models. Since these methodologies are ad-hoc, they require their own explanation, meaning the explanations are often opaque to the user.

3. Explanations can be gamed. Explanations are *generated*. I hope no one would do this, but a neural net designed to explain another neural net could be trained specifically to generate explanations that please users; not explanations inform users. For example, we may wish to deploy a clinical decision support software that generates plain text explanations. A separate AI must be trained to generate explanations from the AI doing the decision support. If user feedback is integrated into the model unchecked, the AI could get to the point where it tells users what they want to hear rather than what they need to hear. More nefarious AI practitioners could specifically design systems to generate explanations that were satisfactory for bypassing regulation, such as those proposed by the FDA for software as a medical device.


## Interpretability

Rather, what we need is models that are interpretable; models whose knowledge can be displayed directly to the decision-maker without a translator. Showing the machine knowledge directly to the decision-maker allows decision-makers to selectively trust the model based on its beliefs.

Again, find a more in-depth discussion of explainabilty and interpretability [here](@/blog/explainable-problems.md).

# Learning from streams

Retraining means waiting. Waiting means things are happening that you do not know about and cannot react to. If you learn from data as they come in you have instant information and are able to react to situations with the most up-to-date knowledge possible. Waiting a week for an updated model is unacceptable in health and defense applications where things can change dramatically without warning.

# Epistemic awareness

Safe AI must be aware of its knowledge, or *epistemically aware*. This means that it knows what it knows and can tell you when it does not. It can tell you under what conditions it is likely to succeed, and under what conditions it is likely to fail. Furthermore, it can tell you when things look weird (anomalies or data entry errors), and identify and characterized overarching world-level changes that affect its performance and understanding of the world. 

In addition to increasing the robustness of an AI, a number of other safety features arise from epistemic awareness.

## Uncertainty quantification

There are really two types of uncertainty: uncertainty associated with the data and uncertainty associated with the model. The model may be very certain that a particular prediction has high variability simply because the data are highly variable; or it may be highly uncertain about a low variance prediction because it just can figure out what is going on. We need to know both of these things. If the natural variance of the data are high, we may need to search for another supporting variable to explain that variance. For example, the height of any human is more variable than the height of a two-year-old human. On the other hand, if the model cannot figure out how to model a prediction, we need to hedge our bets, or to ask the system what extra data it needs to do a better job, or to take some other intervention.

## Anomaly detection

If an AI has an intuitive model of the world it can know when a datum does not adhere to that model. This is anomaly detection. If the underlying model is nonsense, it cannot sensibly detect anomalies. Often 'outlier' and 'anomaly' are confused. An outlier is a datum that lies too far away from the average. The user must decide how far is too far. It also requires that the concept of 'farness' can be applied to the data. This becomes difficult with categorical data (like eye color), or with mixed variable types. Anomalousness is in the eye of the beholder, and if the AI beholds too differently from the user, there will be incompatibilities which lead to failures, distrust, and disuse.

## Novelty detection

Novelty detection is a broader concept than anomaly detection. We think of an anomaly being a one-off observation, while novelty can be an event that changes many observations. For example, a new class arises: our model which is trained to classify images of fruits starts receiving images of a vegetable. Or maybe the interaction between variables changes: the floor of the building in which we run an assay becomes important to prediction because of mold issue. Or maybe the entire world goes crazy because of a global viral pandemic (a rather far-fetched scenario). How will your AI respond? It will break. Tragically.

A safe AI must recognize these situations. It must tell the user what changed, when the change occurred and what affect the change has on its performance.

# Reactive

Detecting when things go wrong is great, but detecting and reacting is better. Reaction can take many forms. Maybe the AI re-composes itself to handle world-breaking events, or perhaps it asks for help. For example, systems meant to track power use behavior pre- and post-global-pandemic can probably be broken into two distinct parts. But it would be good for policy makers to know which model, pre- or post-pandemic, better describes behavior at a given time to help determine the mood of the public at large.

At a smaller scale, a system should be able to ask for information when it is unsure, so it must know which information it needs to improve itself. In this way the AI could direct the data collection process to optimize learning and knowledge. As a patient is admitted to the hospital, health care professionals (HPCs) run tests to ultimately diagnose and treat the patient. At each intervention performed by HPCs the AI could show its belief about the diagnosis, its certainty, and recommend which interventions will help it make the most informed decision.

# Conclusion: Ethically transferring responsibility to people

AI decision support safety is a two-part problem. AI with the above features will be safe in a production-wise sense: it will provide vital knowledge quickly, it will be robust to odd data and events, and it will provide a means of self-defense and self-improvement. But this machine is part of a team. And it is not the key player; it is a humble advisor. So, the machine must not only learn safely, it must communicate safely. People have a host of social learning biases that help them to learn incredibly quickly, but that have caused a lot of problems when interacting with machines. In a future post, I'll discuss these issues.

# Key Points

To be safe, and artificial intelligence must

1. Store its knowledge in a way that is naturally human interpretable
2. Learn from streams of data
2. Be *epistemically aware*, that is be aware of its own knowledge which allows it to
    - Detect anomalous data/events
    - Identify knowledge gaps
    - Characterize its performance
    - Detect world-level changes to the data process
3. Characterize and react to novelty
