res = [[51, 3, 2], [2, 2, 0], [8, 3, 10]]
score = 0
for i in range(3):
    if sum(res[i][:]) == 0 or (res[0][i] + res[1][i] + res[2][i]) == 0:
        continue
    precision = res[i][i] / sum(res[i][:])
    recall = res[i][i] / (res[0][i] + res[1][i] + res[2][i])
    score = score + (2 * precision * recall) / (precision + recall)
print(score / 3)