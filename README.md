# audioKNN-Classifier
This project classifies audio features in the time and frequency domains using MATLAB. Audio features are extracted and classified via the k-Nearest-Neighbors (kNN) algorithm, with dimensionality reduction performed through Principal Component Analysis (PCA) for simplified 3D visualization and minimized information loss.

# Dataset
The dataset includes three classes of sounds (cow, clapping, engine) from the ESC-50 environmental sound classification dataset, representing animal, human (non-vocal), and urban sounds. Each class is split 70/30 for training and testing.

# Features extraction
To optimize feature extraction, a step-length of 20ms is set. 
The window size is tested at different values (30ms, 50ms, 100ms and 500ms) to determine the optimal window size for each class. 
Each sub-struct (containing the training and test sets for each class) is iterated to extract the audio features from each file. 
The function “frequency_features” extracts the characteristics related to the frequency domain: spectrum, centroid, roll-off, and Mel frequency cepstrum (MFCCs). 
The function “timedomainFeats” extracts the characteristics related to the time domain: zero crossing rate, entropy, and average energy. 
The audio features of the training sets are extracted first, followed by those of the test sets.

# PCA 
PCA is applied to the data extracted from the time and frequency features to reduce the number of variables, thereby simplifyng the classification model. 
In the plot, the three sound classes are represented by different colors: cow – red, clapping – green, engine – blue

# kNN 
The kNN training is performed separately on the time domain and frequency domain features, and subsequently, both features are combined. 
For each of the three analyses, labels representing the class affiliations of the training and test data are created.
