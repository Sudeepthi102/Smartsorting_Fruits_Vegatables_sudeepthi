
import os
import pickle
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

DATASET = "dataset"
X, y = [], []
class_names = sorted(os.listdir(DATASET))

for idx, cls in enumerate(class_names):
    cls_path = os.path.join(DATASET, cls)
    for img_name in os.listdir(cls_path)[:50]:
        try:
            img = Image.open(os.path.join(cls_path, img_name)).convert("RGB")
            img = img.resize((64, 64))
            X.append(np.array(img).flatten() / 255.0)
            y.append(idx)
        except:
            pass

X, y = np.array(X), np.array(y)
model = SVC(kernel="linear", probability=True)
model.fit(X, y)

os.makedirs("model", exist_ok=True)
with open("model/fruit_classifier.pkl", "wb") as f:
    pickle.dump((model, class_names), f)

print("Model trained and saved")
