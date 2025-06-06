# ECG-Signal-Classification-For-Heart-Disease-Detection
A Full Stack Web Application that aim to help clients to detect heart condition by analyzing their ECG signals, we used 3 deep learning models alexnet, resnet18 and squeezenet1_1 that classify the heart condition into 3 categories.
A stacking ensmeble model was developed that took the individual outputs and trains on them, before giving the final prediction. It was observed that our ensemble  odel using KNN as meta classifier gave accuracy more than 97% surpassing the accuracies of all individual models. 

We used ECG signals of three categories:
  •	Cardiac Arrhythmia (ARR)
  •	Congestive Heart Failure (CHF) 
  •	Normal Sinus Rhythms (NSR).
  
These signals are obtained from 162 ECG recordings from three PhysioNet databases:
  •	MIT-BIH Arrhythmia Database (96 Recordings) [ARR Signals]
  •	MIT-BIH Normal Sinus Rhythm Database (30 Recordings) [NSR Signals] 
  •	BIDMC Congestive Heart Failure Database (36 Recordings) [CHF Signals].

Applying CWT on these 1D signals will give 2D scalograms that will be stored.(see ecgdataset.zip)

Download all the required modules and run alex3.py.

