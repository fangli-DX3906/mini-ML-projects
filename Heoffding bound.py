import numpy as np
from collections import Counter
import pandas as pd
import random
import matplotlib.pyplot as plt


# verify Hoeffding bound

# A class of coin-flipping
class CoinFlip:
    def __init__(self, num):
        self.num = num

    def flipping(self, times):
        if self.num == 1:
            c = Counter(np.random.randn(times) > 0)
            result = {'Head': c[False], 'Tail': c[True]}
        else:
            result = []
            for i in range(self.num):
                c = Counter(np.random.randn(times) > 0)
                result_ind = {'Head': c[False], 'Tail': c[True]}
                result.append(result_ind)
        self.result = result
        return self.result

    def randomPick(self):
        temp = np.array(range(self.num))
        random.shuffle(temp)
        return temp[0]

    def freqCal(self, result):
        freq_li = list(map(lambda x: x['Head'] / (x['Head'] + x['Tail']), result))
        return np.asarray(freq_li)


# pre-set params
criterion = ['first', 'rand', 'min']
color = ['b', 'g', 'r']
numCoin = 1000
times = 10
repeat = 10000

# 
c1 = CoinFlip(numCoin)
c1_freq = c1.freqCal(c1.flipping(times))
c1_first = c1_freq[0]
c1_rand = c1_freq[c1.randomPick()]
c1_min = np.min(c1_freq)


# in order to improve the efficiency, ret the number to 20000
first = []
rand = []
min = []
for i in range(repeat):
    flip = CoinFlip(numCoin)
    freq = flip.freqCal(flip.flipping(times))
    first.append(freq[0])
    rand.append(freq[flip.randomPick()])
    min.append(np.min(freq))
result = [first, rand, min]

plt.figure(figsize=(18, 6))
for i in range(len(result)):
    plt.subplot(1, 3, i + 1)
    plt.hist(result[i], 11, (0, 1), color=color[i])
    plt.title(criterion[i])
    plt.xlabel('bin')
    plt.ylabel('frequency')
plt.show()

# 
epsilon = np.linspace(0, 0.5, 1000)
hoeffbound = 2 * np.exp(-2 * (epsilon ** 2) * times)
yvalue = []
for i in range(len(result)):
    yvalue.append([])
    freq = np.asarray(result[i])
    for j in epsilon:
        temp = freq[np.bitwise_or(freq > 0.5 + j, freq < 0.5 - j)].shape[0]
        yvalue[i].append(temp / repeat)

plt.figure()
for i in range(len(result)):
    plt.plot(epsilon, yvalue[i], color=color[i])
plt.plot(epsilon, hoeffbound, color='k')
plt.legend(['first', 'rand', 'min', 'Hoeffding Bound'])
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$\mathbb{P}[|\nu-\mu|>\epsilon]$")
plt.title('Different Coins')
plt.show()