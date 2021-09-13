# Transpondancer

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Yuni0217/Transpondancer">
    <img src="https://github.com/Yuni0217/Transpdance/blob/main/Figures/Prototype_gif.gif" alt="Logo">
  </a>
  <br />

  <!-- Badges -->
  <img src="https://img.shields.io/github/repo-size/Yuni0217/Transpdance?style=for-the-badge" alt="GitHub repo size" height="25">
  <img src="https://img.shields.io/github/last-commit/Yuni0217/Transpdance?style=for-the-badge" alt="GitHub last commit" height="25">
  <img src="https://img.shields.io/github/license/Yuni0217/Transpdance?style=for-the-badge" alt="License" height="25">
  <br />
 

  <h3 align="center">Transpondancer</h3>
  <p align="center">
    Transmit, Responce, Dance it.
    <br />
    <a href="https://www.youtube.com/watch?v=kmX7v3Qewd4&t=2s"><strong>View the demo »</strong></a>
    <br />
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Motivation](#motivation)
  * [Framework](#framework)
* [Dataset](#dataset-for-the-deep-learning-model)
* [Solution](#solution)
* [Installation](#installation)
* [Future Work](#future-work)
* [References](#references)

## About the Project
Each dance creates another body-relation-system of knowledge in sensing, anatomically structures, emotional codings of body parts, metaphors, expression and imagination. 
The vocabulary itself is complex and reflects synaesthetic relations of body and memory, historical transformations and body based knowledge, but there is no dance encyclopedia yet. Then how can we name a movement in dance? Moreover, How can AI help in achieving this in a generalized manner?

**Transpondancer** is a tool that automatically generates a textual step-by-step dance guide from any dancing video. In order to achieve this, we have proposed a framework along with a prototype that will be a real product given sufficient data. 

### Motivation
- How to name a movement where each dance creates another body-relation-system of knowledge in sensing, anatomically structures and emotional codings of body parts?
- The Transpondancer address these issues through movement poetics and multilingual body knowledge. Dance Studies, Choreographers, Dancers and everyone who wants to get to know more about dance background, vocabulary and the ways of movement creation will take part in the corresponding process.
- Oral vocabulary that has not been written down will take part and saved from disappearing. It's a fluid dance encyclopedia on the move. 


### Framework
Following this framwork is how one can tackle this challege. Below we are going to breakdown the framework and provide a walk through.
<p align="center">
<img src="https://github.com/Yuni0217/Transpondancer/blob/main/Figures/OnlineClassification.png" alt="System" width="650px">
</p>

**Step1:** Extracting the frames from the dance video dataset. (e.g. a short part of ballet dance, followed by two sliced frames)
<p align="center">
<img src="https://github.com/Yuni0217/Transpdance/blob/main/Figures/ballet.gif" width="140" height="150"> <img src="https://github.com/Yuni0217/Transpdance/blob/main/Figures/balletslice1.png" width="120" height="150"> <img src="https://github.com/Yuni0217/Transpdance/blob/main/Figures/balletslice2.png" width="120" height="150">
</p>

**Step2:** Generating skeleton postures using point light display, or directly implementing [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose). 
<p align="center">
<img src="https://github.com/Yuni0217/Transpdance/blob/main/Figures/balletedited1.png" width="120" height="150"> <img src="https://github.com/Yuni0217/Transpdance/blob/main/Figures/balletedited2.png" width="120" height="150">
</p>

**Step3:** Biological motion perception, or correlation between human movement to textual descriptions. (The first sliced ballet pose is "second arabesque", and the second sliced  ballent pose is "assemble")

**Step4:** Inserting the textual dance movement description into the frame and re-creating the video.
<p align="center">
<img src="https://github.com/Yuni0217/Transpondancer/blob/main/Figures/balletNamed1.png" width="170" height="200"> <img src="https://github.com/Yuni0217/Transpondancer/blob/main/Figures/balletNamed2.png" width="170" height="200">
</p>

## Dataset for the Deep-Learning Model
**Our own dataset** includes 2 [datasets](https://github.com/Yuni0217/Transpondancer/tree/main/Data), one for Ballet movement classification, and the other for Locking movement classification. Upon extracting the respectice Dataset, make sure the files are organized in the format that specifies [here](https://github.com/Yuni0217/Transpondancer/blob/main/src/Ballet/datahandler.py)

Although the number of images that we could collect are limited due to time contraints and resources, we are constantly adding more and more and any new contributions towards the dataset are always welcome.

## Solution
- Finding large amounts of data was and is a great challenge for most of the problems in AI. As it is also the case for Transpondancer, we have collected our own ["dataset"](https://github.com/Yuni0217/Transpondancer/tree/main/Data) of different dance styles.
- Since most of the images are directly taken from the internet, there is a definite need of preprocessing them before passing them to the model. This is done by with the help of [data_handler script](https://github.com/Yuni0217/Transpondancer/blob/main/src/Ballet/datahandler.py) which transforms the images into specified shape and returns batches for both train and validation.
- Finally, we've trained and produced ["Deep-learning models"](https://github.com/Yuni0217/Transpondancer/tree/main/models) from which one can be able to identify dance pose of selected genres or can also be a starting point for future models.
- As some of the models are too big to be uploaded here, they are uploaded to google drive and can be accessed with this [link](https://drive.google.com/drive/folders/1yuKeE1WE-RuAPukYZXMvNR44FFZCTOo4?usp=sharing)
- Here is how the workflow process looks like,
<p align="center">
<img src="https://github.com/Yuni0217/Transpondancer/blob/main/Figures/Workflow.PNG" width=700 height=200>
</p>


## Installation
1. Clone the repo using the following command:
```bash
git clone https://github.com/Yuni0217/Transpondancer.git 
```
2. Create a virtual environment with Python 3.7. (For this step I will assume that you are able to create a virtual environment with `virtualenv`, but in any case you can check an example [here](https://realpython.com/python-virtual-environments-a-primer/).)

3. Install requirements using `pip`:
```bash
pip install -r requirements.txt
```

4. Extract the datasets in the [Data](https://github.com/Yuni0217/Transpondancer/tree/main/Data) folder and run the following command. If you are able to see batches of images in a grid-like view, you are good to go.
```bash
python test.py
```

5 . To start training the model, run the following command. You can always tune the parameters in the `train.py` script
```bash
python train.py
```

## Future Work

- Classfication of movement in an image or a video by following **time series** approach is planned to reduce the error. 

- **Sound classification** will also be added by incorporating sound design tools such as Oscillators, filters, effects, Equalizer (e.g high pass, low pass, notch, etc.) can help    recreate the various sounds attributed to the dance styles.
## References

* [PyTorch](https://github.com/pytorch/pytorch) for training deep-learning models.
* [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) for real-time multi-person keypoint detection for body, face, hands, and foot estimations.
