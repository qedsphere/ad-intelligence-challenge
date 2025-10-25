# Ad Intelligence Challenge

## Overview

At the core of AppLovin's **Axon technology** is a real-time recommendation system that processes thousands of features about users, advertisers, and the ads they see — all to predict which creative, at this very moment, will drive the best result.

This challenge is about diving into that foundation: **how can we extract smarter, more meaningful signals from ad creatives themselves?**

---

## Your Goal

Create a model or prototype that processes ad creatives (images or videos) and extracts novel, high-value features or insights that could feed into a recommendation engine.

### Examples may include:
- Tone or emotional sentiment
- Type of product or content being advertised
- Text, logo, or object recognition
- Audio or visual embeddings
- Any other creative or contextual signals that could inform ad performance prediction

---

## What We're Looking For

### 1. Signal Extraction Insight
- Ability to extract diverse high-value signals from ad creatives
- Signals should be distinct and minimally overlapping
- Clear justification for why these signals could improve a recommendation or ranking model

### 2. Performance
- Processing should be fast (< 5 minutes) and parallelizable
- Clear architecture and rational behind design decisions

### 3. Robustness
- Results should be repeatable and consistent
- Should work on a wide variety of ad types - product ads, app ads, video creatives, and static images

### 4. Creativity
- Demonstrate outside-the-box thinking in what features to extract and how to interpret them

---

## Judging Criteria (weighted)

| Criteria | Weight |
|----------|--------|
| Signal Extraction Insight | 25% |
| Performance | 25% |
| Robustness | 25% |
| Creativity | 25% |

---

## Details

In this folder, `ads.zip` contains both sample image and video advertisements from advertisers across multiple industries, including automotive, fashion, consumer products, insurance, and more. 

**Important notes:**
- Ads are either a `.png` or a `.mp4`
- ⚠️ They do **not** have consistent sizing, frame rates, or resolutions
- Some have clear specific products they promote, whereas others are traditional brand ads promoting the company in general
- Each ad is given an ID, for example, `i0001.png`

You should use these ads to test your feature extraction pipeline.

---

## What Makes a Good Feature?

### Distinctiveness
If features are highly correlated, they add little new information.

### Predictive Power
Even though this challenge doesn't include performance labels, imagine how your feature would correlate with engagement, clickthrough, or purchases. 

**Examples:**
- The presence of a CTA button often drives higher engagement
- Too little action at the beginning of an ad can cause a user to lose interest

### Scalability
The feature should be fast and consistent to compute across millions of ads. 

- ✅ **Good:** "Average motion intensity per frame" can be batch processed
- ❌ **Bad:** "Manually annotated facial expression" is not scalable

---

## Technical Guidelines

- [ ] Process both images and videos (optional: extract frames or audio)
- [ ] Aim for total processing time **under 5 minutes** on the provided dataset
- [ ] Design your system to be parallelizable or batchable
- [ ] Use any open-source model(s) or libraries
- [ ] Output your extracted features in any readable format

---

## Final Thoughts

This challenge is about **seeing ads differently** – transforming pixels, motion, and audio into structured intelligence that could shape how recommendation systems understand creativity.

> Think creatively. Prototype boldly. Surprise us with signals that no one has ever thought of before. And remember, a great feature isn't just data – **it is insight**.

**Good luck!** 🚀
