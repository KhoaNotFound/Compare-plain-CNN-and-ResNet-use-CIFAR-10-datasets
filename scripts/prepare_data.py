#%%
from datasets import load_dataset
from datasets import load_from_disk
from img_classification.config import RAW_DIR
#%%
#load data
dataset = load_dataset("uoft-cs/cifar10")

#inspect schema
print(dataset)
classes = dataset["train"].features["label"].names
print(classes)

#save data to data/raw
dataset.save_to_disk(str(RAW_DIR))
