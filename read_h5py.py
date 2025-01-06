

import h5py
import numpy as np

filename = "2.h5"
data = h5py.File(filename, 'r')

T = np.array(data["grasps/transforms"])
# q = np.array(data["grasps/qualities/gazebo/physical_qualities"])
f = np.array(data["grasps/qualities/Force_closure"])
d = np.array(data["grasps/qualities/Dexterity"])
t = np.array(data["grasps/qualities/Torque_optimization"])

for group in data.keys() :
    print ("Group:" + group)
    print("Data in group: " + str(data[group]))
    if group != "object":
        for dset in data[group].keys():      
            print ("Name of subgroup: " + dset)
            ds_data = data[group][dset]
            print(ds_data)
            # dset is a np array 
            print(len(ds_data))
            if dset != "qualities":
                arr = ds_data[0] # returns the values
                print(arr)
            else:
                #qualities consists of  dexterity, force_closure and torque_optimization
                print(ds_data.keys())
                for d_subset in ds_data.keys():
                    #print score for all grasps (2001 total),
                    print("Scores for: " + str(d_subset))
                    for idx in range(3): #range(len(ds_data[d_subset])):
                        print(ds_data[d_subset][1])
            # print (arr.shape, arr.dtype)
            # print (arr)

