# Transpondancer

How can we name a movement in dance? The movements or vocabulary in dance changes quickly within seconds and there can be quite a lot of them within a few seconds of a dance clip. How do we approach this?

**Transpondancer** is a tool that automatically generates a textual step-by-step dance guide from any dancing video. In order to achieve this, we have proposed a framework by which the problem can be broken into sup-parts and can be solved with the right data and computational power. 

**Objective:** is to generate and print out textual descriptions for each dancing movement in a video. 

**Challenges:** include several points. 1) dance movement is usually in sequential, meaning one movement may cover several essential poses; 2) the same dance movement may have different terminologies or be named under different terms; 3) dance dataset is hard to find, especially for machine-learning model training purposes. 

## Current Method is based on video sliced image classification. 

A bird-view framework is presented below to illustrate the methodology.

<img src="https://github.com/Yuni0217/Transpondancer/blob/main/Figures/OnlineClassification.png" width="250" height="150">

**Step1:** Slicing the dancing video dataset. (e.g. a short part of ballet dance here, followed by two sliced pictures)

<img src="https://github.com/Yuni0217/Transpdance/blob/main/Figures/ballet.gif" width="170" height="170"> <img src="https://github.com/Yuni0217/Transpdance/blob/main/Figures/balletslice1.png" width="120" height="150"> <img src="https://github.com/Yuni0217/Transpdance/blob/main/Figures/balletslice2.png" width="120" height="150">

**Step2:** Generating skeleton postures using point light display, or directly implementing [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). 

<img src="https://github.com/Yuni0217/Transpdance/blob/main/Figures/balletedited1.png" width="120" height="150"> <img src="https://github.com/Yuni0217/Transpdance/blob/main/Figures/balletedited2.png" width="120" height="150">

**Step3:** Biological motion perception, or correlation between human movement to textual descriptions. (The first sliced ballet pose is "second arabesque", and the second sliced ballent pose is "assemble")

<img src="https://github.com/Yuni0217/Transpondancer/blob/main/Figures/balletNamed1.jpeg" width="170" height="200"> <img src="https://github.com/Yuni0217/Transpondancer/blob/main/Figures/balletNamed2.jpeg" width="170" height="200">


## Video Data

**Video Scene Elements** include static and dynamic objects. Static objects refer to dancing props like a phone, a door, un umbrella, etc., as well as background props like a table and a door. Dynamic objects include a person and his/her bodyparts or an animal. 

**Video Scene Structure** include the spatial position (meaning if the human object is in the left, the right or other places of the scenes), and the human actions such as head movements, hand actions and other body poses. 

## Dataset for ML model training

**Our own dataset** includes 2 dataset, one for Ballet movement classification, and the other for Locking movement classification. Ballet dataset further contains 100 pictures for Arabesque, 86 pictures for Battement, 

## References

* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for real-time multi-person keypoint detection library for body, face, hands, and foot estimations.
