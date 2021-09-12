# Transpondancer

Each dance creates another body-relation-system of knowledge in sensing, anatomically structures, emotional codings of body parts, metaphors, expression and imagination. 
The vocabulary itself is complex and reflects synaesthetic relations of body and memory, historical transformations and body based knowledge, but there is no dance encyclopedia yet. Then how can we name a movement in dance? Moreover, How can AI help in achieving this in a generalized manner?

**Transpondancer** is a tool that automatically generates a textual step-by-step dance guide from any dancing video. In order to achieve this, we have proposed a framework along with a prototype that will be a real product given sufficient data. 
<p align="center">
<img src="https://github.com/Yuni0217/Transpdance/blob/main/Figures/Prototype_gif.gif">
</p>


## Framework
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

## Alternative
Finding large amounts of data was and is a great challenge for most of the problems in AI. As it is also the case for Transpondancer, we have collected our own ["dataset"](https://github.com/Yuni0217/Transpondancer/tree/main/Data) of different dance styles and produced different offline ["Deep-learning models"](https://github.com/Yuni0217/Transpondancer/tree/main/models) from which one can be able to identify dance pose of selected genres. We have also trained multiple models for each genre that can be either used directly or can be a starting point for future models.
## Dataset for the Deep-Learning Model

**Our own dataset** includes 2 [datasets](https://github.com/Yuni0217/Transpondancer/tree/main/Data), one for Ballet movement classification, and the other for Locking movement classification. Upon extracting the respectice Dataset, make sure the files are organized in the format that specifies [here](https://github.com/Yuni0217/Transpondancer/blob/main/src/Ballet/datahandler.py)

Although the number of images that we could collect are limited due to time contraints, we are constantly adding more and more and new contributions are always welcome via pull requests.

## Future Works

**Image or video classification enhancement** is planned to improve the accuracy of movement classification. 

**Sound classification** will also be added. Incorporating sound design tools such as Oscillators, filters, effects, Equalizer (e.g high pass, low pass, notch, etc.) can help recreate the various sounds attributed to the dance styles. Further research into sound engineering is required to be able to implement this. 

## References

* [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) for real-time multi-person keypoint detection library for body, face, hands, and foot estimations.
