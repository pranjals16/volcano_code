# -*- coding: utf-8 -*-
__author__ = "Pranjal Singh"
"""
This module prepares data from raw files and segregates into
training, test and validation files.

Example:
    python data_preparation.py
"""

import glob
from sklearn.model_selection import train_test_split

######################
# Global Variables #
######################
BASE_PATH = "C:/Users/pranjal/Desktop/kettle/model_data/"
SCIENCE_PATH = BASE_PATH + "science/*.txt"
TECH_PATH = BASE_PATH + "technology/*.txt"
CITIES_PATH = BASE_PATH + "cities/*.txt"
GLOBAL_PATH = BASE_PATH + "global/*.txt"
OUTPUT_PATH = "../data/"
POS_CLASS_IDX = 3


def read_dir(path, all_data, label, class_idx):
    for file_name in glob.glob(path):
        with open(file_name, "rb") as f:
            all_data.append(" ".join(f.readlines()))
            # Taking care of Positive anf Negative Label.
            if class_idx == POS_CLASS_IDX:
                label.append(1)
            else:
                label.append(0)
    return all_data, label


def write_files(x_list, y_list, file_type):
    print "Current File Size: ", len(x_list)
    with open(OUTPUT_PATH + file_type + "_x.txt", "w") as f:
        for item in x_list:
            text_item = item.replace('\n', ' ').replace('\r', '')
            f.write(text_item + "\n")
    with open(OUTPUT_PATH + file_type + "_y.txt", "w") as f:
        for item in y_list:
            f.write(str(item) + "\n")


######################
# Main #
######################
def main():
    all_data = []
    label = []
    path_list = [SCIENCE_PATH, CITIES_PATH, GLOBAL_PATH, TECH_PATH]
    for i in range(0, len(path_list)):
        all_data, label = read_dir(path_list[i], all_data, label, i)

    x_train, x_test, y_train, y_test = train_test_split(all_data, label, test_size=0.20, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)
    write_files(x_train, y_train, "train")
    write_files(x_test, y_test, "test")
    write_files(x_val, y_val, "val")


if __name__ == "__main__":
    main()
