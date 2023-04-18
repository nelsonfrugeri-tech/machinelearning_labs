from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# features
# long hair | short leg | makes ruff (1 for DOG, 0 for PIG)
pig1 = [0, 1, 0]
pig2 = [0, 1, 1]
pig3 = [1, 1, 0]
dog1 = [0, 1, 1]
dog2 = [1, 0, 1]
dog3 = [1, 1, 1]

animals = [pig1, pig2, pig3, dog1, dog2, dog3]
results = [1, 1, 1, 0, 0, 0]

# Training
model = LinearSVC()
model.fit(animals, results)

# Testing
test1 = [[1,1,1], [1,1,0], [0,1,1]]
expected = [0, 1, 1]
predict = model.predict(test1)

print("Hit rate: ", accuracy_score(expected, predict) * 100)