import os
import json
import random

#Go over all the grasp files, split them in two lists: train and test
#Dump them in a json dict

grasp_root_dir = "../DA2/data/grasps"
split_root_dir = "../DA2/data/splits"

train_grasp_files = []
test_grasp_files = []
split_percentage = 0.8 
for filename in os.listdir(grasp_root_dir):
    print(filename)
    #Randomly assign to train or test
    if random.random() < split_percentage:
        train_grasp_files.append(filename)
    else:
        test_grasp_files.append(filename)
print(len(train_grasp_files), len(test_grasp_files))
split_dict = {"train": train_grasp_files, "test": test_grasp_files}
with open(os.path.join(split_root_dir, "split.json"), "w") as f:
    json.dump(split_dict, f)
    print("Split saved to split.json")