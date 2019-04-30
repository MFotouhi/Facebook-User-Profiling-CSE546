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

#input_path = "C:\\Users\\bhagatsanchya\\Desktop\\UWash\\Machine_Learning\\Project\\python_code\\data\\public-test-data"
#output_path = "C:\\Users\\bhagatsanchya\\Desktop\\Myoutput"
#maj_gender_predic_str="female"
##majority_age="xx-24"
#extrovert="3.48685789474"
#neurotic="2.73242421053"
#agreeable="3.58390421053"
#conscientious="3.44561684211"
#openness="3.90869052632"

"""
    Predictions using text
"""

# read text input
print("Reading input")
input = rp.read_text(input_path)

# age predictions using text
print("Making age prediction using text")
age_predictions_text = tcf.make_predictions(input, "naive_bayes", "age", "text")

# personality trait predictions
personality_predictions = {}
for personality_trait in tfe.personality_traits:
    print("Making", personality_trait, "prediction using liwc")
    personality_predictions[personality_trait] = tcf.make_predictions(input, "linear_regression", personality_trait, "liwc")

# gender predictions
print("Making gender prediction using text")
gender_predictions_text = tcf.make_predictions(input, "naive_bayes", "gender", "text")

'''
    Predictions using image
'''

print("Making gender prediction using image")
print(input_path)
gender_predictions_image = icc.make_predictions(input_path, "gender")
