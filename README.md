# TalkingDataAdtrackingFraudDetection


Script to predict fraud clicks in ads.
Working with this kind of dataset is hard because it is a very large dataset and the classes are very unbalanced. Here, we have 2 classes, 0 and 1, indicating id the click resulted in a download of the app or not. The script process the file in small chunks and  apply a filter creating a new file with a similar number of observations in both classes. Then, this balanced file is read and it is made an exploraty data analysis and feature engineering in order to get a better fit of the model. Once it is done, it is trained a XGBoost and a LightGBM algorithms to predict if the download was executed or not.


The dataset is composed by the following variables:


*ip:* click IP address;

*app:* application id referred by the ad;

*device:* dispositive type ID;

*os:* operational system version;

*channel:* channel of the announcement editors;

*click_time:* register and time of the click;

*attributed_time:* if the app was downloaded, it shows the time of the download;

*is_attributed (Target):* target feature indicating if the app was downloaded or not.



This repository contains:

**TalkingDataAdTrackingFraudClickDetection.R** - the r script to generate the datasets, train the models and make the predictions.

**TalkingDataAdTrackingFraudClickDetection.html** - the html report with the code and comments.

It will be generated by the code:

**databalanced.csv** - the file with balanced data generated by the scrip. As this step take a long time, I already put it here.

**predictionsLGBM.csv** - the file with the click_id and predictions.


