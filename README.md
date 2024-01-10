# Multi-channels change points detection

<img src="/plots/data_set_1_Root.jpg" alt="Alt text">

## Introduction
This repo offers the EMULLR algorithm which is an algorithm for detectiong change points in a multi-channel time series data set using a log likelihood ratio test.<br>
More information can be found in our [paper](Change_points_detection_in_multichannels_time_series__Final_Project.pdf).<br>
The algorithm was developed by Nadav Elyakim (<27nadav@gmail.com>) and Or Katz (<katzor20@gmail.com>)  as part of the MSc graduation project at Reichman University in collaboration with VIA Transportation.

## Running the model
In order run the model:<br>
```
!clone https://github.com/ok123123123/Multi_Dim_CP_Detection.git

!pip install -r requirements.txt
from src.model import ChangePointDetection
model = ChangePointDetection(channels=channels)
change_points_list = model.fit()
```

## [Notebook](https://github.com/ok123123123/Multi_Dim_CP_Detection/blob/main/EMULLR.ipynb)
A notebook that demonstrates use EMULLR algorithm on a real data set taken from the *** company and then 
 presents various experiments carried out during the development of the algorithm on generated data set.<br> 

## Thanks for visiting


