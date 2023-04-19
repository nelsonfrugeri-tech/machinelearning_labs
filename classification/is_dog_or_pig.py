from kit_learn import Machine

mac = Machine()

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
mac.fit(animals, results)

# Test 01
expected = [0, 1, 0]
predict = mac.predict([dog3, pig3, dog1])

print("Hit rate: %.2f" % (mac.hit_rate_in_percent(expected, predict)))

# Test 02
expected = [1, 1, 0]
predict = mac.predict([pig1, pig2, dog2])

print("Hit rate: %.2f" % (mac.hit_rate_in_percent(expected, predict)))

# Test 03
expected = [0, 1, 0, 1, 0, 1]
predict = mac.predict([dog2, pig3, dog1, pig2, dog2, pig1])

print("Hit rate: %.2f" % (mac.hit_rate_in_percent(expected, predict)))
