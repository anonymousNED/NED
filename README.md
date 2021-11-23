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

## Video preprocessing
To train or test the method on a specific subject, first create a folder for this subject and place the video(s) of this subject into a **"videos"** subfolder. The training videos for the 6 Youtube actors used in our experiments can be downloaded from [here](https://drive.google.com/drive/folders/1sH3r0XIwLrLYSCpxu6zqqtE2v7MAXHVa?usp=sharing), while the test videos for the same actors are available [here](https://drive.google.com/drive/folders/1OyMePR3jitM4qYArmJ04xTPtraGHMU_T?usp=sharing).

For example, for testing the method on Tarantino's clip, a structure similar to the following must be created:
```
Tarantino ----- videos ----- Tarantino_t.mp4
```
Under the above structure, there are 3 options for the video(s) placed in the "videos" subfolder:
1. Use this footage to train a neural face renderer on the actor (e.g. use the training video one of our 6 Youtube actors, or a footage of similar duration for a new identity).
2. Use it as test footage for this actor and apply our method for manipulating his/her emotion.
3. Use it only as reference clip for transferring the expressive style of the actor to another subject.

To preprocess the video (face detection, segmentation, landmark detection, 3D reconstruction, alignment) run:
```bash
./preprocess.sh <celeb_path> <mode>
```
- ```<celeb_path>``` is the path to the folder used for this actor
- ```<mode>``` is one of ```{train, test, reference}``` for each of the above cases respectively

After successfull execution, the following structure will be created:

```
<celeb_path> ----- videos -----video.mp4 (e.g. "Tarantino_t.mp4")
                   |        |
                   |        ---video.txt (e.g. "Tarantino_t.txt", stores the per-frame bounding boxes, created only if mode=**test**)
                   |
                   --- images (cropped and resized images)
                   |
                   --- full_frames (original frames of the video, created only if mode=**test** or mode=*reference*)
                   |
                   --- eye_landmarks (created only if mode=**train** or mode=*test*)
                   |
                   --- eye_landmarks_aligned (same as above, but aligned)
                   |
                   --- align_transforms (similarity transformation matrices, created only if mode=**train** or mode=*test*)
                   |
                   --- faces (segmented images of the face, created only if mode=**train** or mode=*test*)
                   |
                   --- faces_aligned (same as above, but aligned)
                   |
                   --- masks (binary face masks, created only if mode=**train** or mode=*test*)
                   |
                   --- masks_aligned (same as above, but aligned)
                   |
                   --- DECA (3D face model parameters)
                   |
                   --- nmfcs (NMFC images, created only if mode=**train** or mode=*test*)
                   |
                   --- nmfcs_aligned (same as above, but aligned)
                   |
                   --- shapes (detailed shape images, created only if mode=**train** or mode=*test*)
                   |
                   --- shapes_aligned (same as above, but aligned)
```
## Manipulate the emotion on a test video
Download our pretrained manipulator from [here](https://drive.google.com/drive/folders/1fLAsB2msBcLnRJWlixXt-hJ8FeX3Az6T?usp=sharing) and unzip the checkpoint.

Then, assuming that preprocessing (in **test** mode) has been performed for a given video (see above), we can manipulate the expressions of the celebrity in this video by one of the following 2 ways:

### Label-driven manipulation
Select one of the 7 basic emotions (happy, angry, surprised, neutral, fear, sad, disgusted) and run :
```bash
python manipulator/test.py --celeb <celeb_path> --checkpoints_dir ./manipulator_checkpoints --trg_emotions <emotions> --exp_name <exp_name>
```
- ```<celeb_path>``` is the path to the folder used for this actor's test footage (e.g. "./Tarantino")
- ```<emotions>``` is one or more of the 7 emotions. If one emotion is given, e.g. ```--trg_emotions happy```, all the video will be converted to happy, whereas for 2 or more emotions, such as ```--trg_emotions happy angry``` the first half of the video will be happy, the second half angry and so on.
- ```<exp_name>``` is the name of the sub-folder that will be created under the <celeb_path> for storing the results.

### Reference-driven manipulation
In this case the reference video should first be preprocessed (see above) in **reference** mode. Then run:
```bash
python manipulator/test.py --celeb <celeb_path> --checkpoints_dir ./manipulator_checkpoints --ref_dirs <ref_dirs> --exp_name <exp_name>
```
- ```<celeb_path>``` is the path to the folder used for this actor's test footage (e.g. "./Tarantino")
- ```<ref_dirs>``` is one or more of the reference videos. In particular, the path to the "DECA" sublfolder has to be given. As with labels, more than one paths can be given, in which case the video will be transformed sequentially according to those reference styles.
- ```<exp_name>``` is the name of the sub-folder that will be created under the <celeb_path> for storing the results.
