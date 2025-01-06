#!/usr/bin/python3
print("Hello World")
import json
import os

#take current directory
current_dir = os.getcwd()
print(current_dir)

for filename in os.listdir(current_dir + "/grasps/"):
    print(filename)
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)
            print(data)
    else:
        continue