"""
Rank Selection
==============

Authors: Federico Raimondo, Kaustubh Patil

License: BSD 3 clause

"""
from opnmf.selection import rank_permute
from opnmf.logging import configure_logging
import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################
# set up logging
configure_logging('INFO')

##############################################################################
# Load IRIS dataset
iris = sns.load_dataset("iris")
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris[features].values.T


##############################################################################
# Find rank. In this example we are bounded by the number
# of features (4)
min_components = 1
max_components = 4

result = rank_permute(X, min_components, max_components)

good_ranks, tested_ranks, errors, random_errors, estimators = result

##############################################################################
# Plot the results

plt.figure()
plt.title('Rank selection on IRIS dataset')
plt.plot(tested_ranks, random_errors, label='permuted')
plt.plot(tested_ranks, errors, label='original')

good_errors = errors[good_ranks - min_components]

plt.plot(good_ranks, good_errors, label='selected', marker='o', c='r',
         ls='None')

plt.xticks(tested_ranks)
plt.xlabel('# Components')
plt.ylabel('Error')
plt.legend()
plt.show()
