import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = './dataset/data.pickle'
MODEL_DIR = './models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.p')

with open(DATA_PATH, 'rb') as f:
    dataset = pickle.load(f)

data = np.asarray(dataset['data'])
labels = np.asarray(dataset['labels'])

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(f"Acurácia nos testes: {score * 100:.2f}%")

os.makedirs(MODEL_DIR, exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"Modelo salvo em {MODEL_PATH}")