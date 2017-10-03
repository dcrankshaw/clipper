import sys
import os
import numpy as np
import pickle

from kernel_svm_model import KernelSvmModel

VGG_FEATURE_VEC_SIZE = 4096
TRAINING_DATA_SIZE = 2000

def train_svm_model(output_path):
	model = KernelSvmModel()

	training_data = 10 * np.random.rand(TRAINING_DATA_SIZE, VGG_FEATURE_VEC_SIZE)
	training_labels = np.random.randint(2, size=TRAINING_DATA_SIZE)

	model.train(training_data, training_labels)
	output_file = open(output_path, "w")
	pickle.dump(model, output_file)
	output_file.close()

	print("Trained and saved model!")

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage is 'python kernel_svm_trainer.py <model_output_path>")
		raise

	model_output_path = sys.argv[1]

	train_svm_model(model_output_path)