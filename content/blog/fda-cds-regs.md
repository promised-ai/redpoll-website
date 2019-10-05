+++
title = "Training deep networks to skirt FDA regulation"
date = 2019-10-08
template = "post.html"
draft = false

[taxonomies]
categories = ["ai", "ethics"]

[extra]
author = "Baxter Eaves"
subheading = "Or: Concerns with weak language in the FDA's guidance on clinical decision support software"
image = "healthcare-hospital-lamp.jpg"
theme = "light-transparent"
+++

The FDA recently released (September 29th 2019) a guidance on the regulation of clinical decision support (CDS) software. The guidance focuses in part on what software falls under the definition of a medical "device" and is therefore subject to regulation. In this post, I will discuss how certain terminology could be attacked to avoid regulation, and how the psychology surrounding *decision bases* should impact a software's regulation status. I argue that there being a health care professional between patient and software is a poor reason not to regulate a software product.

# The FDA Guidance

[The guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)

# Decision Basis

The primary factor to determining whether a CDS systems is or is not a device is whether it is intended to be the primary decision maker. In order not to be labeled as a device the software must be 

> [...] intended to enable [health care professionals] to independently review the basis for the recommendations presented by the software so that they do not rely primarily on such recommendations, but rather on their own judgment, to make clinical decisions for individual patients.

What is a basis? From line 254 of the guidance:

> In order to describe the basis for a recommendation, regardless of the complexity of the software and whether or not it is proprietary, the software developer should describe the underlying data used to develop the algorithm and should include plain language descriptions of the logic or rationale used by an algorithm to render a recommendation.

These are not satisfying constraints on modern AI. To what degree do we have to describe logic or rationale? What is rationale? Is "The patient is 24 year old, so they might enjoy an anticoagulant" sufficient rationale? To what degree does the description have to align with the mathematical workings of the algorithm? Is there a distinction between *algorithm* and *model*? Can I invoke the stochastic gradient descent optimization algorithm to have my software labeled as an unregulated non-device?

If we look at the complex, black box AI we have today, there is really no way to plainly and completely describe the logic. We can say something like "Input data X, Y, and Z were important factors to this flag". But because many of these models are nonsensical, any notion of *important* applies to only one particular model, and does not reflect what actually is important, which is misleading. An audit trail could be derived that follows the inputs' activation of various weights in a neural network, but again, this is nonsensical; it does not help the physician to scrutinize the decision.

## A safe decision basis

Safety is only an issue when it's an issue. If a CDS is designed only to recommend things like preventative measures or non-invasive diagnostics, there isn't much of a concern for safety. The FDA does touch on this a bit, but they frame it in terms of the criticality (FIXME: check language) of the patient. Software for use with patients who are not in a very fragile state of health will be unregulated, because I assume you have to mess up a lot worse to hurt someone who is relatively healthy. Ok. Yikes. But lets say we're recommending medical treatment. We're recommending drugs or invasive diagnostics or risky treatments. Regardless of the state of the patient, doing the wrong thing here can go very badly.

## Training a machine learning model to skirt regulation

Now that providing basis is a path to regulation-free sales, and since there is no acceptable definition of what a sufficient basis is, a basis in large part can be manufactured. And it's easy to attack this weak language.

The language is weak because what it means is left to judgement. It it not put to unambiguous words. A good decision basis is then like pornography: As [Supreme Court Justice Potter Stewart said](https://www.law.cornell.edu/supremecourt/text/378/184) "I know it when I see it". Given any explanation, a practitioner will know whether that explanation is an acceptable decision basis under the guidelines. As software developers, we observe the explanations our software generates and the judgment of the end user (acceptable or unacceptable). Now all we need to do is learn the nebulous definition that links explanations to judgements. That unknown definition is an unknown function from inputs (explanations) to output (judgements)... We have a machine learning problem ideal for black box classification frameworks.

The attack is simple: use feedback from healthcare professionals (HCPs) and the FDA itself to train a recurrent neural network to avoid regulation by generating superficially acceptable decision bases. We attach the recurrent neural network to the end of whatever black box is making the recommendations. We do the initial training on bases generated by inexpensive lay people. We then get feedback from HPCs telling us which explanations are acceptable; then retrain. If we haven't decided to shove the thing into production and take our chances with the FDA, we could then seek approval or feedback from FDA personnel who will certainly point out examples of poor explanation. Our training data are then the lay explanations, HPC feedback, and FDA feedback. If our training data are rich enough, we have trained a neural network to produce explanations that pass scrutiny but that are completely brittle and senseless. If we've done really well, we also have explanations that HPCs favor, which means we're selling software HPCs favor, which means we're probably selling more software.

But really we've just trained a stupid brittle system that HPCs love. And because our stupid brittle system is not *intended* to be used as a primary decision-maker, when it goes wrong and is successful at tricking an HPC into accepting a bad recommendation, it's the HPC who will take the fall.

This of course seems very unethical but this is really only because I've been forthright in my use of the word "attack". This "attack" is actually how much of AI is brought to market. Jiggle things until they *work*, for whatever definition of work. Often "working" means telling someone what they want to hear. Nearly this exact "attack" has already been perpetrated by at least one large (unnamed) corporation acting in healthcare -- whereby fake data were used to train a black box model until the training physicians were satisfied. The real-world results weren't good, but fortunately physicians were able to pick up on it and prevent the machine from doing real harm. Detecting and defending against dangerously stupid AI should not be the job of HPCs. We're counting our lucky stars that nothing went wrong then, but we're not learning from that experience.

No one seems interested in working out the ethics of industrial AI (post on this forthcoming). People believe they must accept that AI works in mysterious ways; that black box decision making is something they just have to come to terms with. It is thought that if we place unreasonable expectations on AI, like absolute and fundamental transparency, we will stifle innovation. And imagine all the world-changing things we will miss out on!

So my question to the FDA is: considering that medical error is already the [3rd leading cause of death in the US](https://www.hopkinsmedicine.org/news/media/releases/study_suggests_medical_errors_now_third_leading_cause_of_death_in_the_us), how many eggs do we need to break to make a CDS omlet?

## Toward a safe definition of decision basis

A decision basis should be considered sufficient if it conveys, without ambiguity:

1. Which inputs are relevant to the decision and the degree of relevancy
3. All alternative predictions/decisions, their likelihood, and certainty
4. The relationship between the inputs, via the model parameters, to the outputs (decision or prediction)
    - The shape of the prediction/decision function as a function of all relevant inputs
    - Ideally the HPC should be able to manipulate inputs and see the effect. This would help to appropriately modulate trust and discover failure modes.


<!-- # Inconsistent quality standards across physical and software devices

Both a blood pressure monitor and a pacemaker are medical devices, but they are medical devices for different reasons. The pacemaker is a medical device in part because it makes decisions regulating a patient's heart rate without HPC consult. A blood pressure monitor is a medical device because it is designed to collect data. Both devices have to adhere to strict reliability standards. A blood pressure monitor should give accurate and precise measurements for a known number of uses, and pacemaker shouldn't crap out. Saying that a non-device CDS should have no reliability standards because it is the doctor's job to recognize bad decisions is smiliar to asserting that a blood pressure monitor should be allowed to give horribly inaccurate measurement because it is the HPC who decides to operate that device.

-->

# Wrapping up

A software's regulation status in wrapped up in whether it provides sufficient basis for its decision. A basis is an explanation. Machine explanations are not necessarily auditable nor are they necessarily sensible. Given enough approval/disapproval from regulators, researchers would be trained in how to train black box AI to generate superficial bases that pass clinical scrutiny but are brittle in the real world. There are two solutions:

1. Develop a definition of decision basis that revolve around linking inputs to output via the full structure of the machine knowledge (hard).
2. Label all CDS software as a device and regulate it (easy).
