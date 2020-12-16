import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances, accuracy_score

# the constants for creating the data
n_tot = 200
ntr = 50
nts = n_tot-ntr
nc = 10

X, y = load_digits(n_class=nc, return_X_y=True)

# divide into training and testing
Xtr = X[:ntr, :]
ytr = y[:ntr]
Xts = X[ntr:(ntr+nts), :]
yts = y[ntr:(ntr+nts)]

# # the hints for coding:
# pdists = pairwise_distances(X, Y, metric="hamming")*X.shape[1]}
# np.round(np.random.rand(nc, rcode_len))

# ------------------------------------------------------------------------------------------

min_hammings = []
accs = []
for rcode_len in np.arange(int(np.ceil(np.log2(nc)))+2, 40):

    for ii in range(5):  # or 20, or even 5 gives answer in the range

        # ¤¤¤¤¤¤¤¤¤¤ create random codewords for the classes ¤¤¤¤¤¤¤¤¤¤

        random_code = np.round(np.random.rand(nc, rcode_len))
        unique_codewords = True

        # because the codeword matrix is created randomly there is no guarantee that it would be suitable

        # let's first prune out repeating columns
        random_code = np.unique(random_code, axis=1)

        # prune out also columns consisting of all zeros or all ones
        colsums = np.sum(random_code, axis=0)
        cols_to_keep = np.where((0!=colsums) & (colsums!=nc))[0]
        random_code = random_code[:, cols_to_keep]

        # this one checks if all the codewords (rows) are unique;
        if len(np.unique(random_code, axis=0)) != nc:
            unique_codewords = False

        if unique_codewords:

            # ¤¤¤¤¤¤¤¤¤¤ calculate smallest hamming distance ¤¤¤¤¤¤¤¤¤¤

            pdists = pairwise_distances(random_code, random_code, metric="hamming")*random_code.shape[1]
            np.fill_diagonal(pdists, np.nan)  # diagonal is full of zeroes; we don't want those to affect the results!
            min_hammings.append(np.nanmin(pdists))

            # ¤¤¤¤¤¤¤¤¤¤ classify with ecoc shceme ¤¤¤¤¤¤¤¤¤¤

            # binary classifiers and their predictions
            all_preds = np.zeros((len(yts), random_code.shape[1]))
            for jj in range(random_code.shape[1]):

                # make the binary labels
                ybin = np.zeros(len(ytr))
                oneclasses = np.where(random_code[:, jj] == 1)[0]
                ybin[[indx for indx in np.arange(ntr) if ytr[indx] in oneclasses]] = 1

                ybints = np.zeros(len(yts))
                ybints[[indx for indx in np.arange(len(yts)) if yts[indx] in oneclasses]] = 1

                # train classifier on the new binary labels, get its predictions
                classifier = Perceptron()
                classifier.fit(Xtr, ybin)
                preds = classifier.predict(Xts)
                all_preds[:, jj] = preds

            # then get the final predictions by checking the Hamming distances to the codewords
            pred_pdists_to_code = pairwise_distances(random_code, all_preds, metric="hamming")*rcode_len
            preds = np.argmin(pred_pdists_to_code, axis=0)
            accs.append(accuracy_score(yts, preds))

accs = np.array(accs)
min_hammings = np.array(min_hammings).astype(int)

unique_hammings = np.unique(min_hammings)
mean_accs = []
std_accs = []
for hamming in unique_hammings:
    mean_accs.append(np.mean(accs[min_hammings == hamming]))
    std_accs.append(np.std(accs[min_hammings == hamming]))

mean_accs = np.array(mean_accs)
std_accs = np.array(std_accs)

print('unique_hammings ', np.shape(unique_hammings))
print(unique_hammings)

print("accuracy at 1 minimum hamming vs 10 minimum hamming:")
#print("%5.2f+-%4.2f  %5.2f+-%4.2f " % (mean_accs[unique_hammings==1], std_accs[unique_hammings==1],
#                                       mean_accs[unique_hammings==10], std_accs[unique_hammings==10]))
print("difference: %4.2f" % (mean_accs[unique_hammings==[10]]-mean_accs[unique_hammings==[1]]))

#
# --------------------------------------------------------------------------------
# plotting the results, doesn't directly answer the question but is interesting
#

import matplotlib.pyplot as plt
plt.figure()

plt.scatter(min_hammings, accs)
plt.plot(unique_hammings, mean_accs, c="g", label="mean")
plt.xlabel("minimum hamming distances")
plt.ylabel("accuracy")
plt.legend()

plt.tight_layout()
plt.show()

