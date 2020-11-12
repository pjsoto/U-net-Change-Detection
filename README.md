# U-net-Change-Detection-Classifier
The current project contains the scripts to perform a change detection classification for remote sensing data, specifically for deforestation detection in two Brazilian biomes, the Amazon rainforest(Brazilian Legal Amazon) and Brazilian savannah (Cerrado). Regarding the U-net architecture used here, we follow the structure of the one proposed in [1] but adapted to the conditions of our datasets by using a different size of samples at the input of the network.

# Pre-requisites
1- Python 3.7.4

2- Tensorflow 1.14.0

# Train and Test
To run the training and testing stages execute python Main_Script_Executor_Tr_ARO_Ts_ARO_APA_CMA.py. From such a script, all main functions will be executed. Specifically, in this case, the code is prepared to train the classifier on Amazon-RO and test in Amazon-RO, Amazon-PA, and Cerrado-MA. 

# References
[1]Ronneberger, Olaf; Fischer, Philipp; Brox, Thomas (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation". arXiv:1505.04597 [cs.CV].
