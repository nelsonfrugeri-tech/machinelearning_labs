from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# features
# long hair | short leg | makes ruff (0 for DOG, 1 for PIG)
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

# Test 01
expected = [0, 1, 0]
predict = model.predict([dog3, pig3, dog1])

print("Hit rate: %.2f" % (accuracy_score(expected, predict) * 100))

# Test 02
expected = [1, 1, 0]
predict = model.predict([pig1, pig2, dog2])

print("Hit rate: %.2f" % (accuracy_score(expected, predict) * 100))

# Test 03
expected = [0, 1, 0, 1, 0, 1]
predict = model.predict([dog2, pig3, dog1, pig2, dog2, pig1])

print("Hit rate: %.2f" % (accuracy_score(expected, predict) * 100))
