# Talk Script: Longtao Jiang (continual learning, Q2–Q3)

Deck: `talk_longtao.pdf`. Target: **5 minutes** of talk (≈650 words at ~130
words/minute), then 10 minutes of Q&A. Slides 12–17 are backup for questions.
**Bold** words are the ones to stress.

---

## Slide 1 · Title (0:10)

Hi, I'm Longtao. Our project studies how to pick frames from long videos. My
part is about one question: **can continual learning learn the right mixing
weight?**

## Slide 2 · Team and my part (0:15)

We are a team of three. Jianxiang set up the problem and the data pipeline and
led the overall study. Makhmud evaluated question answering. I ran and analysed
the **continual learning experiments**, which are questions two and three of our
study.

## Slide 3 · Background: two clues, one weight (0:40)

A vision-language model can only see about **eight frames** of a long video. A
method called VSI picks them using two clues for every second: whether the
question's objects appear, and whether the subtitle matches the question. It
mixes them with a weight lambda and keeps the top eight. VSI uses **0.3 for
every question**. In this real example, 0.3 misses the answer moment, but 0.8
hits it.

## Slide 4 · My part: learning the weight (0:35)

We found that the best lambda depends on the question type: text questions want
**0.8 to 0.9**, visual questions **0.3 or less**. So the idea is to train a small
predictor that outputs lambda for each question. New question types keep
arriving, so it learns them **one after another**, without forgetting the old
ones. That is **continual learning**. I asked two things: can it learn lambda,
and does it forget? And if the gain is small, why?

## Slide 5 · Setup: four types in sequence (0:40)

We have four question types, learned in order: text perception, visual
perception, text relation and visual relation. We try three orders and three
random splits, so **nine runs** per method. We compare six methods: naive
training, EWC which protects important weights, LwF which imitates the old
model, replay which reuses old questions, nearest neighbours, and joint training
as a reference. We measure final accuracy and forgetting.

The key part is the baselines. Besides the default 0.3, we also use **one lambda
tuned on the training set**. This changes the story, as you'll see.

## Slide 6 · Q2: results over 9 runs (0:40)

Here are the results. Every method beats the default by **four to five points**,
in all nine runs. That looks like a clear win. But compared with one tuned
lambda, the best methods, LwF and joint, gain only **0.6** points. Naive, EWC and
replay are about equal to it. And no method reaches one lambda per question
type.

## Slide 7 · Q2: the right baseline matters (0:30)

This slide shows every run. On the left, against the default, all dots are
above zero. On the right, against the tuned value, they sit **around zero**. So
if we only compared with the default, we would overstate how much learning
helps. A learned weight must be compared with a **tuned constant**.

## Slide 8 · Q3: one weight for every type (0:40)

Why is the gain so small? Here is the average lambda the predictor gives to each
question type after each training stage. The triangles are the best values for
each type. The four lines **almost overlap** at about 0.6. The predictor did not
learn to give text questions a higher weight. It just learned a **better global
value**, which is why it only ties the tuned constant.

## Slide 9 · Q3: why, and forgetting (0:40)

We see two reasons. First, the signal is **sparse**: 62 percent of questions give
the same result for any lambda, so almost all learning comes from the remaining
third. Second, the predictor has **no information about the question type**; it
only sees the score curves, which look similar across types.

This also explains forgetting. All types get similar weights, so learning a new
one overwrites little. LwF and joint forget about **half** as much as naive, but
only in six of nine runs.

## Slide 10 · Takeaways from Q2–Q3 (0:25)

So: continual learning beats the default in every run. But it only ties a tuned
constant, because it collapses to one weight for all types. Forgetting is small,
and anti-forgetting methods help only weakly. Next, we would give the predictor
the question type, or start from one weight per type and learn per-question
corrections.

## Slide 11 · Thank you (0:05)

Thank you. Happy to take questions.

**Total ≈ 5:00.**

---

## Q&A preparation (10 minutes)

Jump to the backup slide in brackets when it helps.

**Q: How exactly is the predictor trained?** [Slide 12]
It outputs probabilities over eleven values of lambda and uses their weighted
average. Most questions give the same recall for every lambda, so "best lambda"
labels carry little signal. Instead we train it to rank the annotated answer
seconds high in the mixed score. 50 epochs per type, Adam, learning rate 0.001.

**Q: What are EWC, LwF and replay, in one sentence each?**
EWC adds a penalty for changing weights that mattered for old tasks. LwF asks
the new model to give the same outputs as the old model on new data. Replay
keeps up to 200 old questions and trains on them again.

**Q: How much room is left?** [Slide 13]
From the default to one tuned value is 4.5 points, one value per type adds 1.4,
and the best value per question adds 8.9 more. Our predictor captures roughly
the first step.

**Q: Can you show forgetting in more detail?** [Slides 14 and 15]
The recall matrices show each type's recall after each training stage. The rows
change little, which matches the small forgetting values: Naive 0.029, LwF and
Joint 0.015.

**Q: Why is Joint not much better than the continual methods?**
Joint trains on all types together, so it has no forgetting problem. It reaches
the same 0.668 as LwF. That tells us the limit is not forgetting but what the
predictor can learn from these inputs.

**Q: How did you choose the tuned constant?**
On the training split only: the value with the best average training recall. It
is 0.7 in two splits and 0.9 in one. No test data.

**Q: Did learning ever look worse than the default?** [Slide 16]
Yes. Early on, we trained on the average lambda but tested on the most likely
value. The predictor put its probability on zero and one, so testing picked the
two worst extremes, and learned weights looked three to five points worse. After
the fix, they beat the default.

**Q: Why not add the question text as input?** [Slide 17]
We tried a 768-dimensional question embedding. With only about 500 training
questions it overfit and made test recall lower. Curve statistics plus 44
features describing the two curves worked best.

**Q: Why does kNN do worst?**
kNN copies the best lambda of similar questions. But the best lambda is noisy
and often tied, since many questions are flat, so copying it from neighbours
works poorly.

**Q: Would more question types change the conclusion?**
Possibly. With only four types and similar learned weights there is little
conflict. A longer sequence with more different types would test forgetting
more seriously.
