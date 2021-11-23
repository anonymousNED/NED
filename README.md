## Neural Emotion Director (NED)
This is the official Pytorch implementation of NED.

<img src="imgs/teaser.gif" width="100%" height="100%"/>

> **Neural Emotion Director: Speech-preserving semantic control of facial expressions in “in-the-wild” videos**<br>
> <br>
>
> **Abstract:** *In this paper, we introduce a novel deep learning method for photo-realistic manipulation of the emotional state of actors in ``in-the-wild'' videos. The proposed method is based on a parametric 3D face representation of the actor in the input scene that offers a reliable disentanglement of the facial identity from the head pose and facial expressions. It then uses a novel deep domain translation framework that alters the facial expressions in a consistent and plausible manner, taking into account their dynamics. Finally, the altered facial expressions are used to photo-realistically manipulate the facial region in the input scene based on an especially-designed neural face renderer. To the best of our knowledge, our method is the first to be capable of controlling the actor’s facial expressions by even using as a sole input the semantic labels of the manipulated emotions, while at the same time preserving the speech-related lip movements. We conduct extensive qualitative and quantitative evaluations and comparisons, which demonstrate the effectiveness of our approach and the especially promising results that we obtain. Our method opens a plethora of new possibilities for useful applications of neural rendering technologies, ranging from movie post-production and video games to photo-realistic affective avatars.*

## Getting Started
Clone the repo:
  ```bash
  git clone https://github.com/anonymousNED/NED
  cd NED
  ```  

### Requirements
Create a conda environment, using the provided ```environment.yml``` file.
```bash
conda env create -f environment.yml
```
Activate the environment.
```bash
conda activate NED
```

### Files
1. Follow the instructions in [DECA](https://github.com/YadiraF/DECA) (under the *Prepare data* section) to acquire the 3 files ('generic_model.pkl', 'deca_model.tar', 'FLAME_albedo_from_BFM.npz') and place them under "./DECA/data".
2. Fill out the [form](https://docs.google.com/forms/d/e/1FAIpQLScyyNWoFvyaxxfyaPLnCIAxXgdxLEMwR9Sayjh3JpWseuYlOA/viewform) to get access to the [FSGAN](https://github.com/YuvalNirkin/fsgan)'s pretrained models. Then download 'lfw_figaro_unet_256_2_0_segmentation_v1.pth' (from the "v1" folder) and place it under "./preprocessing/segmentation".
