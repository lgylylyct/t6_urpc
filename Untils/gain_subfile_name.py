import os

folder_path = r"E:\SegExp\DataSet\002_spilt_mask_nii"
train_txt_path = r"E:\SegExp\DataSet\label_data\train.txt"
val_txt_path = r"E:\SegExp\DataSet\label_data\val.txt"
test_txt_path = r"E:\SegExp\DataSet\label_data\test.txt"

# Get the names of all files in the folder
file_names = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
#file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# Get the surnames of the files
surnames = [f.split(".")[0] for f in file_names]

# Calculate how many sub-directory names to select
train_num = int(len(surnames) * 3 / 5)
val_num = int(len(surnames)* 1 / 5)

# Select three-fifths of the sub-directory names
train_num_names = surnames[:train_num]
val_num_names = surnames[train_num:train_num+val_num+1]
test_num_names = surnames[train_num+val_num+1:]



# Save the surnames to a file
with open(train_txt_path, "w") as file:
    for surname in train_num_names:
        file.write(surname + "\n")

with open(val_txt_path, "w") as file:
    for surname in val_num_names:
        file.write(surname + "\n")

with open(test_txt_path, "w") as file:
    for surname in test_num_names:
        file.write(surname + "\n")



