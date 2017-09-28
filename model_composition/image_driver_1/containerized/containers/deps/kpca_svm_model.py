from __future__ import division
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn import svm, preprocessing
from sklearn import decomposition
from sklearn.externals import joblib
import numpy as np
import pdb
import sys

class KpcaSvmModel:
	
	def __init__(self, num_components, gamma):
		"""
		Initializes the model

		Parameters
		----------
		num_components : int
		    The number of principal components to be extracted
		    during the kernel_pca training phase (see the kernel_pca() method)

		gamma : float
		    The scaling value used in the kernel function phi(x)
		    during the kernel_pca training phase (see the kernel_pca() method)
		"""

		self.num_components = num_components
		self.gamma = gamma
		self.trained = False

	def train(self, training_inputs, training_labels):
		"""
		Trains the model

		Parameters
		----------
		training_inputs : np.ndarray
			An m-by-n matrix of training inputs (m vectors of n features each)

		training_labels : np.ndarray
			An m-by-1 matrix of training labels
		"""

		self.training_inputs = np.copy(training_inputs)

		pc, kernel_matrix = self.kernel_pca(training_inputs, self.gamma, self.num_components)
		classifier = self.fit_svm(pc, kernel_matrix, training_labels)

		self.principal_components = pc
		self.classifier = classifier
		self.trained = True


	def evaluate(self, inputs):
		"""
		Evaluates the model on a matrix of inputs

		Parameters
		----------
		inputs : np.ndarray
			An m-by-n matrix (m vectors of n features each)

		Returns
		----------
		np.ndarray
			An m-by-1 matrix of predicted labels
		"""

		if not self.trained:
			print("Cannot evaluate an untrained model!")
			raise

		projected_inputs = self.transform_and_project_inputs(inputs)
		outputs = self.classifier.predict(projected_inputs)

		return outputs

	def transform_and_project_inputs(self, inputs):
		"""
		Transforms inputs based on the kernel function and projects them 
		onto the principal component space for SVM classification

		Parameters
		----------
		inputs : np.ndarray
			An m-by-n matrix (m vectors of n features each)
		"""
		kernel_matrix = []
		num_inputs = inputs.shape[0]
		for i in range(0, num_inputs):
			input_dists = self.training_inputs - inputs[i,:]
			input_dists = input_dists.dot(input_dists.T)
			input_dists = input_dists.diagonal()

			kernel_dists = exp(-self.gamma * input_dists)
			kernel_dists = np.asarray(kernel_dists, dtype=np.float64)

			kernel_matrix.append(kernel_dists)

		kernel_matrix = np.asarray(kernel_matrix, dtype=np.float64)
		projected_inputs = kernel_matrix.dot(self.principal_components)

		return projected_inputs		

	def kernel_pca(self, inputs, gamma, n_components):
		"""
		Performs kernel pca on an m-by-n data matrix
		with kernel function phi(x) = e^(-gamma * x)

		Parameters
		----------
		inputs : np.ndarray
		    An m-by-n matrix (m vectors of n features each)

		gamma : float
		    Scaling value used in the kernel function phi(x)

		n_components : int
		    The number of principal components to be extracted

		Returns
		----------
		(np.ndarray, np.ndarray)
		    A tuple containing two matrices. The first matrix
		    is of shape m-by-`n_components` - this is the matrix
		    of principal eigenvectors. The second matrix is 
		    the kernel matrix and is of shape m-by-n
		"""

		sq_dists = pdist(inputs, 'sqeuclidean')
		mat_sq_dists = squareform(sq_dists)
		k = exp(-gamma*mat_sq_dists)
		k_prime = self.center_kernelized(k)
		eigvals, eigvecs = eigh(k)
		principal_components = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
		return principal_components, k_prime

	def pca(self, inputs, n_components):
		pca_executor = decomposition.PCA(n_components)
		pca_executor.fit(inputs)
		return pca_executor.components_

	def center_kernelized(self, kmat):
		"""
		Centers a matrix in kernel space. See definition of K'
		in https://en.wikipedia.org/wiki/Kernel_principal_component_analysis.

		Parameters
		----------
		kmat : np.ndarray
		    An uncentered kernel matrix

		Returns
		----------
		np.ndarray
		    The kernel matrix (K')
		"""

		n = kmat.shape[0]
		one_n = np.ones((n,n))/n
		kmat = kmat - one_n.dot(kmat) - kmat.dot(one_n) + one_n.dot(kmat).dot(one_n)
		return kmat

	def fit_svm(self, principal_components, kernel_matrix, labels):
		"""
		Fits an SVM to inputs that have been processed
		by PCA

		Parameters
		----------
		principal_components : np.ndarray
		    The principal component matrix obtained by invoking the
		    `kernel_pca()` or `pca()` methods with raw feature inputs

		kernel_matrix : np.ndarray
		    An m-by-n kernel matrix obtained by invoking the `kernel_pca()`
		    method with raw feature inputs
		"""
		classifier = svm.SVC()
		kpca_inputs = kernel_matrix.dot(principal_components)
		classifier.fit(kpca_inputs, labels)
		return classifier