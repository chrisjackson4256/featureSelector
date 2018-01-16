'''

This code takes in a data file and returns the 
most relevant features.  

The code should work with any csv-type file as 
long as the file name, delimiter type and the 
existence of a header are specified.

The user has the choice to normlaize/standardize
their feature set (this is highly recommended if
the features are not commensurate.)

The user has the option to use one of two 
techniques (or both) for feature selection:
	1) univariate filtering (where only features 
	   that have a p-value < 0.001 are selected)
	2) recursive reature elimination (where the 
	   algorithm recursively eliminates the least
	   significant features)

'''
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_regression, SelectKBest, chi2, RFE
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# clear the screen
os.system('cls' if os.name == 'nt' else 'clear')

# little function to replace empty strings
def replace_empty(x, default):
	if x == '':
		return default
	else:
		return x

# Ask the user for some input: file name, delimiter and presence of header
data_file = input("Please enter the name of the data file which " +\
	              "[default='linear_regression_challenge.csv']: ").strip()
data_file = replace_empty(data_file, "linear_regression_challenge.csv")

delim = input("\nPlease specify the delimiter used in the file [default='|']: ")
delim = replace_empty(delim, "|")

header = input("\nDoes the data contain a header? [y/N] ").strip().lower()

if header == 'y':
	header = 'infer'
else:
	header = None

# Import the data
print()
print("Importing data...")
try:
	data = pd.read_csv(data_file, sep=delim, header=header)
except FileNotFoundError:
	print()
	print("File not found.  Please try again.")
	sys.exit()

# let the user know the size/shape of the data
print()
print("-" * 50)
print("Your data has {} rows and {} columns.".format(data.shape[0], data.shape[1]))
print("-" * 50)

# ask the user if they would like to view the "head" of the dataframe
print()
view_data = input("Would you like to view the first five rows of your data? [Y/n] ").strip().lower()
view_data = replace_empty(view_data, 'y')

if view_data == 'y':
	print()
	print("-" * 50)
	print(data.head())
	print("-" * 50)

# find out which column is the target variable
print()
target_col = input("Which column is the target variable? [default=0] ").strip().lower()
target_col = replace_empty(target_col, 0)

# make sure target_col is of the correct type
try:
	target_col = int(target_col)		# column name is an integer
except ValueError:
	pass								# column name is a string

# separate the features from the target variable
X_feat = data.drop(target_col, axis=1)
y_feat = data[target_col]

# ask whether they wish to normalize/standardize the data
print()
normalize_data = input("Would you like to normalize/standardize the features in the " +\
	                   "data (such that all values lie between 0 and 1)? [Y/n] ").strip().lower()
normalize_data = replace_empty(normalize_data, 'y')

# normalize the data using scikit-learn's "MinMaxScaler"
if normalize_data == 'y':
	print()
	print("Normalizing data...")
	min_max_scaler = MinMaxScaler()
	X_feat = pd.DataFrame(min_max_scaler.fit_transform(X_feat))

	print()
	print("-" * 50)
	print("Here's a fresh look at the features: ")
	print()
	print(X_feat.head())
	print("-" * 50)

# univariate feature selection
print("\n\n===========================================")
print("Time to find the most significant features!")
print("===========================================")

uni_feat = input("\n\nWe will start with a simple univariate filtering. " +\
	        "If you have a very large dataset, you may wish to skip this.  Proceed? [Y/n] ").strip().lower()
uni_feat = replace_empty(uni_feat, "y")

if uni_feat == 'y':
	print()
	print("Performing univariate feature filtering...")

	# using scikit-learn's "f_regression" method which computes the "F-score" and "p-value"
	f_values = f_regression(X_feat, y_feat)

	# we will consider a feature "significant" if it has a p-value < 10^{-3}
	significant_count = 0
	significant_idx = []
	for i in range(f_values[0].shape[0]):
	    if f_values[1][i] < 1e-3:
	        significant_count += 1
	        significant_idx.append(i)
	        
	significant_idx = np.array(significant_idx)

	print()
	print("-" * 50)
	print("There are {} features which have p-values < 0.001.".format(significant_count))
	print("-" * 50)

	print()
	print("-" * 50)
	print("The column names of those features are: ", list(significant_idx))
	print("-" * 50)

# recursive feature elimination
print("\n\nNext, we consider a more scalable approach for feature selection when " +\
	  "datasets become very large - 'Recursive Feature Elimination'.\n")

rfe_feat = input("How many features would you like to select?  (If you ran the univariate selection part " +\
	             "of this code, you can choose to use the same number of features found there by just hitting " +\
	             "RETURN). Please enter your number now: ").strip()
rfe_feat = replace_empty(rfe_feat, significant_count)
rfe_feat = int(rfe_feat)

print("\nPerforming recursive feature elimination (this may take a few minutes)...")

# perform the recursive feature elimination 
estimator = LinearRegression()
num_features = rfe_feat
remove_features_step = 0.2

selector = RFE(estimator,
               n_features_to_select=num_features, 
               step=remove_features_step)

selector = selector.fit(X_feat, y_feat)

selected = selector.ranking_ == 1

# get the column indices selected by algorithm
idxs_selected_rfe = selected.nonzero()[0]

# check the overlap with the univariate filtering
if uni_feat == 'y':
	idxs_intersect_rfe = np.intersect1d(list(significant_idx), list(idxs_selected_rfe))

print()
print("-" * 50)
print("Selected features: ", list(idxs_selected_rfe))
print("-" * 50)
print()
if uni_feat == 'y':
	print("-" * 50)
	print("Intersection with univariate approach: ", list(idxs_intersect_rfe))
	print("-" * 50)
	print()
	print("-" * 50)
	print("Percentage overlap with univariate approach: " + str(np.round(len(idxs_intersect_rfe) / len(significant_idx) * 100., 1)) + "%")
	print("-" * 50)
	print()
































