import pickle

from drivers_pkg import benchmarking
from drivers_pkg.models import model
from drivers_pkg.models.image.kpca_svm_model import KpcaSvmModel

class VggSVM(model.ModelBase):

	def __init__(self, ks_model_path):
		model.ModelBase.__init__(self)

		ks_model_file = open(ks_model_path, "rb")
		self.ks_model = pickle.load(ks_model_file)
		ks_model_file.close()

	def predict(self, inputs):
		"""
		Given a list of vgg feature vectors encoded as numpy arrays of data type
		np.float32, outputs a corresponding list of image category labels
		"""
		return self.ks_model.evaluate(inputs)
	
	def benchmark(self, batch_size=1, avg_after=5):
		benchmarking.benchmark_function(self.predict, benchmarking.gen_vgg_svm_classification_inputs, batch_size, avg_after)