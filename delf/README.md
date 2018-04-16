# DELF Implementation

extract_features.py takes list of paths of images, and saves extracted DELF features in data dir. 

recog_match_images.py is intended for generating a submission for Recognition challenge, and csvmaker.py is intended for the generating a submisison for Retrieval, using DELF feature extraction.

#### Child Dir
- delf_recog_test : Testing image and feature data of delf implemntation for recognition challenge. 


#### Note
- **recog_match_images.py and csvmaker.py are modified versions of [match_images.py from DELF examples.](https://github.com/tensorflow/models/blob/master/research/delf/delf/python/examples/match_images.py)**
- **extract_features.py is a modified version of [extract_features from DELF examples.](https://github.com/tensorflow/models/blob/master/research/delf/delf/python/examples/extract_features.py)** 

