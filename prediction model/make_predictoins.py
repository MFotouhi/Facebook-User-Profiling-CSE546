import sys
import os
current_file_location = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_location, '../')) # add reference of parent directory

import read_input as rp
import classifiers.text.text_classifier as tcf
import classifiers.text.text_feature_extractor as tfe
import classifiers.relation.rel_nb_classifier as rnb
import classifiers.image.image_cnn_classifier as icc
import classifiers.ensemble as ens


#check and store the input output path
if len(sys.argv) < 3:
    exit()
input_path = sys.argv[1]
output_path = sys.argv[2]
