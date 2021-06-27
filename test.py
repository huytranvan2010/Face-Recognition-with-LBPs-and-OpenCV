import numpy as np
names = ['anh', 'em', 'oi', 'toi', 'anh', 'anh', 'toi', 'toi','em', 'day']

(names, counts) = np.unique(names, return_counts=True)
names = names.tolist()

print(type(names))
print(names)
print(type(counts))
print(counts)

print(names.index('toi'))

print(np.random.choice(range(10), size=3, replace=False))

from sklearn.preprocessing import LabelEncoder
labels = ["anh", 'em', 'oi', 'toi']

le = LabelEncoder()
labels = le.fit_transform(labels)

print("Integer labels: ", labels)

print(le.inverse_transform([0, 0, 1]))
