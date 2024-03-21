# cs131-final


Alyssa Grand Jete          |  Chuyi Arabesque          |   Elisse Arabesque
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/echow626/cs131-final/blob/main/gifs/alyssa.gif)  |  ![](https://github.com/echow626/cs131-final/blob/main/gifs/chuyi.gif) | ![](https://github.com/echow626/cs131-final/blob/main/gifs/elisse.gif)

[Google Drive](https://drive.google.com/drive/folders/107MIEJkrakvlAM3cyD9mJ3sgYNzF0Z98?usp=sharing) with some of the intermediate results and data used


* `archive`: Old files and attempts at parts of the project, moved here for organizational reasons
* `checkpoints`, `dance_checkpoints`: Model weight checkpoints from the pose estimation model and the movement classifier model
* `dance_images`: Images gathered from the following YouTube videos
    * CorbinHolooway1 - https://www.youtube.com/watch?v=PAepvwuWoRs
    * CorbinHolooway2 - https://www.youtube.com/watch?v=PAepvwuWoRs
    * AvaGuirl - https://www.youtube.com/watch?v=YUmkT_lWWVM
    * BridgetWilde - https://www.youtube.com/watch?v=LKCZCtcO1Do
    * BalletClass - https://www.youtube.com/watch?v=M0GMx0vk8Ec
    * JuaKim - https://www.youtube.com/watch?v=Hg2IKQnvlsY
    * TomokaSato - https://www.youtube.com/watch?v=ICcwOsQOaow
    * JohnCrim - https://www.youtube.com/watch?v=QWKaJIQ2ATU
    * ShogoHayami - https://www.youtube.com/watch?v=Ag31VgFNx6o
    * AlexandraLamm - https://www.youtube.com/watch?v=8_EAO0-MRTs
    * PreciousAdams - https://www.youtube.com/watch?v=YT95D6leFNI
    * SemyonChudin - https://www.youtube.com/watch?v=xZdhG63_dfM
    * EnzoSan - https://www.youtube.com/watch?v=abWgBvJDx_A
    * RyanLenkey - https://www.youtube.com/watch?v=2QDta2DWR4g
    * MarianelaNunez - https://www.youtube.com/watch?v=0SWtbwvhsas
    * EricPoor - https://www.youtube.com/watch?v=lRsLP9sadG8
    * JoakimVisnes - https://www.youtube.com/watch?v=boNHS6qClqY
    * Arabesque - https://www.youtube.com/watch?v=E9EVPDiW4uM
* `dance_predictions`: Short videos of us and our friends processed through the OpenPose Colab
* `json_data`: Json version of the MPII Human Dataset annotations sourced from [this tutorial](https://github.com/ilovepose/fast-human-pose-estimation.pytorch)'
* `openpose_json`, `openpose_output`: Dance dataset processed through the OpenPose Colab. Kept separate from `dance_predictions` on purpose.
* `plots`: Some of the plots created for our paper

Code Files:
* `cs131_final_project_model_1 copy.ipynb`, `cs131_final_project_model_1.ipynb`: PoseEstimator Model. The one more up to date is the `copy` version.
* `dance_model copy.ipynb`, `dance_model.ipynb`: MovementClassifier Model. Same as the PoseEstimator Model, `dance_model copy.ipynb` is more up to date
* `dance_preprocess.ipynb`: Dance images preprocessing

Other Single Data Files:
* `dance_label.txt`: Text file mapping Dance Dataset image frames to movement classifier
* `mpii_human_pose.csv`: CSV version of MPII Annotations sourced from [Kaggle](https://www.kaggle.com/datasets/nicolehoelzl/mpii-human-pose-data/data), includes category and activity labels 
