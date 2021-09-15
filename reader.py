import multirank
import numpy as np
import csv

file = open("EU_Transport.csv", "r")
reader = csv.reader(file)
list_csv = []
for row in reader:
    list_csv.append(row)

transport = np.zeros(7492500).reshape(37,450,450)

for row in list_csv:
    l = int(row[0]) - 1
    r = int(row[1]) - 1
    c = int(row[2]) - 1
    transport[l][r][c] = float(row[3])

for layer in range(37):
    transport[layer] += transport[layer].T

print(multirank.multirank(transport, 3, 1, 1))
