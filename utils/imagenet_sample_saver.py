import os
import random


def copy_chosen(source, destination, samples_n=50, cnt=-1):
    for filename in random.sample(os.listdir(current_class_folder), samples_n):
        file_path = os.path.join(current_class_folder, filename)
        os.system(f"cp {file_path} {os.path.join(base_dist_path, dirname)}")
    print(f"Done with {dirname} \t #{cnt}")


base_source_path = ""
base_dist_path = ""

classes_n = len(os.listdir(base_source_path))
print(f"Total of {classes_n} classes")

i = 1
threads = []
for dirname in os.listdir(base_source_path):
    class_dist = os.path.join(base_dist_path, dirname)
    if not os.path.exists(class_dist):
        os.mkdir(class_dist)
    current_class_folder = os.path.join(base_source_path, dirname)
    copy_chosen(current_class_folder, class_dist, 50000 // classes_n, i)
    i += 1
