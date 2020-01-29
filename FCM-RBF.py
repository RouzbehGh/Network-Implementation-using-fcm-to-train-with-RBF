import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def membershipmatrix(V, m, nd, xd, p):
    U = np.zeros([m, nd])
    for i in range(nd):
        # calculate distance of sample i with m cluster centers
        d = np.zeros(m)
        for j in range(m):
            d[j] = np.sqrt((xd[i, 0] - V[j, 0]) ** 2 + (xd[i, 1] - V[j, 1]) ** 2)

        # calculate membership sample i in each cluster center
        for j in range(m):
            add = 0
            for k in range(m):
                add = add + (d[j] / d[k]) ** (2 / (p - 1))
            U[j, i] = 1 / add
    return U


def fcmclustering():
    #  _________________ Start FCM clustering  _________________

    max_epoch = 200  # number iteration of fcm

    # step1: Initialize m cluster centers
    V = np.zeros([m, num_dim])
    for i in range(m):
        for j in range(num_dim):
            V[i, j] = low[j] + (high[j] - low[j]) * np.random.rand()

    for epoch in range(max_epoch):

        # step2: Update the membership matrix U using the cluster center V
        U = membershipmatrix(V, m, num_train, x_train, p)

        # step3: Update the fuzzy cluster centers V using the membership matrix U
        for j in range(m):
            add1 = 0
            add2 = 0
            for i in range(num_train):
                add1 = add1 + U[j, i] ** p * x_train[i, :]
                add2 = add2 + U[j, i] ** p
            V[j, :] = add1 / add2

    return V, U
    #  _________________ End FCM clustering  _________________


def g(Xk, Vi, Ci):
    XV = np.zeros([num_dim, 1])
    XV[:, 0] = Xk - Vi
    return np.exp(-radi*XV.transpose().dot(np.linalg.inv(Ci)).dot(XV))


def idmax(y, nd):
    id_a = 0
    a = y[0]
    for j in range(1, nd):
        if y[j] > a:
            id_a = j
            a = y[j]
    return id_a


#==========================  Main  ============================

#_______________________  Menu  _________________________
print('The parts of the exercise are as follows')
print('1: Investigate the effect of parameters')
print('2: Center radius is 0.1 and Too many number of clusters')
print('3: Center radius is 1 and number of clusters is 3')
sec = int(input('Choose a part please: '))

if sec == 1:
    m = int(input('Please, enter the number of clusters: '))
    radi = float(input('Please, enter center radius: '))
elif sec == 2:
    m = int(input('Please, enter the number of clusters: '))
    radi = 0.1
else:
    m = 3   # the number of clusters
    radi = 1

print('\n ************************************ \n')
print('The list of data is as follows: ')
print(' 1: 2clstrain1200')
print(' 2: 4clstrain1200')
print(' 3: 5clstrain1500')
print(' 4: TA Tests')

choose = int(input('Choose a sample please: '))

# ______________ read data of file __________________________
if choose == 1:
    samples = pd.read_csv('2clstrain1200.csv', header=None)
    num_class = 2
    con = -1
elif choose == 2:
    samples = pd.read_csv('4clstrain1200.csv', header=None)
    num_class = 4
    con = 4
elif choose == 3:
    samples = pd.read_excel('5clstrain1500.xlsx', header=None)
    num_class = 5
    con = 5
else:
    samples = pd.read_excel('', header=None)
    num_class = 1
    con = 1

samples = samples.values
[num_sample, num_dim] = np.shape(samples)
num_dim = num_dim - 1

low = []
high = []
for i in range(num_dim):
    low.append(min(samples[:, i]))
    high.append(max(samples[:, i]))

p = 2  # parameter of fcm: fuzzy index

#  _________________ calculate number of train samples and test samples
num_train = int(np.round(num_sample*0.7))
num_test = num_sample-num_train

#___________________ create train samples and test samples randomly ______________

x_train = np.zeros([num_train, num_dim])  # initialize train set
y_train = np.zeros(num_train, dtype=int)               # initialize labels of train set
Y = np.zeros([num_train, num_class])        # initialize output neural network


x_test = np.zeros([num_test, num_dim])    # initialize test set
y_test = np.zeros(num_test, dtype=int)                 # initialize labels of test set

index_train = np.random.choice(num_sample, num_train, replace=False)
j = 0
k = 0
for i in range(num_sample):

    if any(i == index_train):
        # sample i-th put in train set
        x_train[j, :] = samples[i, 0:num_dim]

        y_train[j] = samples[i, num_dim]
        if y_train[j] == con:
            y_train[j] = 0

        Y[j, y_train[j]] = 1
        j += 1
    else:
        # sample i-th put in test set
        x_test[k, :] = samples[i, 0:num_dim]

        y_test[k] = samples[i, num_dim]
        if y_test[k] == con:
            y_test[k] = 0
        k += 1

#_____________ run fcm ______________________
V, U = fcmclustering()

#_____________ calculate matrix Ci ____________
C = np.zeros([m, num_dim, num_dim])
for j in range(m):
    Ci = 0
    add = 0
    vector = np.zeros([num_dim, 1])
    for i in range(num_train):
        vector[:, 0] = x_train[i, :]-V[j, :]
        multi = vector.dot(vector.transpose())
        Ci = Ci + U[j, k]**p * multi
        add = add + U[j, k]**p
    Ci = Ci/add
    C[j, :, :] = Ci

# _________ calculate matrix G, it is output of first layer _________
G = np.zeros([num_train, m])
for i in range(m):
    for k in range(num_train):
        G[k, i] = g(x_train[k, :], V[i, :], C[i, :, :])

#__________ calculate weights matrix W ______________
W = np.linalg.inv(G.transpose().dot(G)).dot(G.transpose()).dot(Y)

#  _________________  calculate accuracy for train samples  _________

y_NN = G.dot(W)    # output NN for train samples

y_train_e = np.zeros(num_train)
term2 = 0
for i in range(num_train):
    # determine label of sample i
    y_train_e[i] = idmax(y_NN[i, :], num_class)
    term2 = term2 + np.abs(np.sign(y_train[i] - y_train_e[i]))

accuracy_train = 1 - term2/num_train

#  _________________  calculate accuracy for test samples  _________________

# calculate matrix G', it is output of first layer for test samples
G_p = np.zeros([num_test, m])
for i in range(m):
    for k in range(num_test):
        G_p[k, i] = g(x_test[k, :], V[i, :], C[i, :, :])

y_NN = G_p.dot(W)    # output NN for test samples

y_test_e = np.zeros(num_test)
term2 = 0
for i in range(num_test):
    # determine label of test sample i
    y_test_e[i] = idmax(y_NN[i, :], num_class)
    term2 = term2 + np.abs(np.sign(y_test[i] - y_test_e[i]))

accuracy_test = 1 - term2/num_test

#  ________________________ Show result  _________________
print('result with number of cluster m= ', m, ' and radius= ', radi)
print('-------------------------------------------------------------')
print('accuracy train= ', accuracy_train)
print('accuracy test= ', accuracy_test)

#  ________________________ plot result  _________________

leg = []
index_miss = []
for k in range(num_class):
    flg = 0
    index_correct = []

    for i in range(num_test):
        if y_test[i] == y_test_e[i] and y_test[i] == k:
            index_correct.append(i)
            flg = 1
        elif k == 0 and y_test[i] != y_test_e[i]:
            index_miss.append(i)
    if k == 0:
        plt.plot(x_test[index_miss, 0], x_test[index_miss, 1], '*', markersize=10)
        leg.append('missing')
    if flg == 1:
        plt.plot(x_test[index_correct, 0], x_test[index_correct, 1], '+')
        leg.append('class' + str(k))

plt.plot(V[:, 0], V[:, 1], 'pk')
leg.append('cluster centers')
plt.title('m='+str(m)+' and radius= '+str(radi))
plt.legend(leg)
plt.show()

#______________ plot boundary of clustering _______________

# generate new samples
num_new = 100
X = np.linspace(low[0], high[0], num_new)
Y = np.linspace(low[1], high[1], num_new)

x_new = np.zeros([num_new*num_new, num_dim])
k = 0
for j in X:
    for i in Y:
        x_new[k, :] = np.array([i, j])
        k += 1

# calculate number of new samples
num_new = x_new.shape[0]

# calculate membership matrix new samples using centers Vi
U = membershipmatrix(V, m, num_new, x_new, p)

# determine cluster of each new sample
cluster = np.zeros(num_new)
for i in range(num_new):
    # determine cluster of sample i
    cluster[i] = idmax(U[:, i], m)

# plot boundary of clustering
leg = []
for k in range(m):
    flg = 0
    index_cluster = []
    for i in range(num_new):
        if cluster[i] == k:
            index_cluster.append(i)
            flg = 1

    if flg == 1:
        plt.plot(x_new[index_cluster, 0], x_new[index_cluster, 1], '.')
        leg.append('cluster' + str(k))

plt.plot(V[:, 0], V[:, 1], 'pk')
leg.append('cluster centers')
plt.title('m='+str(m))
plt.legend(leg)
plt.show()
