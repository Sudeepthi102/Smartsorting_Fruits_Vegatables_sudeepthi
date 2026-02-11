
import os, random
from PIL import Image
import matplotlib.pyplot as plt

DATASET = "dataset"
cls = random.choice(os.listdir(DATASET))
img = random.choice(os.listdir(os.path.join(DATASET, cls)))
path = os.path.join(DATASET, cls, img)

Image.open(path).show()
print("Class:", cls)
