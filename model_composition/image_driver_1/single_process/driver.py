import argparse
import numpy as np
import json

from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from single_proc_utils import DriverBase, driver_utils
from models import lgbm_model, vgg_feats_model, vgg_svm_model, inception_feats_model

VGG_FEATS_MODEL_NAME = "vgg_feats"
INCEPTION_FEATS_MODEL_NAME = "inception_feats"
SVM_MODEL_NAME = "svm"
LGBM_MODEL_NAME = "lgbm"


GPU_CONFIG_KEY_VGG_FEATS = "vgg_feats"
GPU_CONFIG_KEY_INCEPTION_FEATS = "inception_feats"

########## Setup ##########

def get_heavy_node_configs(batch_size, allocated_cpus, vgg_gpus=[], inception_gpus=[]):
	vgg_config = driver_utils.HeavyNodeConfig(model_name=VGG_FEATS_MODEL_NAME,
											  input_type="floats",
											  allocated_cpus=allocated_cpus,
											  gpus=vgg_gpus,
											  batch_size=batch_size)	

	inception_config = driver_utils.HeavyNodeConfig(model_name=INCEPTION_FEATS_MODEL_NAME,
											  		input_type="floats",
											 		allocated_cpus=allocated_cpus,
											  		gpus=inception_gpus,
											  		batch_size=batch_size)

	svm_config = driver_utils.HeavyNodeConfig(model_name=SVM_MODEL_NAME,
											  input_type="floats",
											  allocated_cpus=allocated_cpus,
											  gpus=[],
											  batch_size=batch_size)

	lgbm_config = driver_utils.HeavyNodeConfig(model_name=LGBM_MODEL_NAME,
											   input_type="floats",
											   allocated_cpus=allocated_cpus,
											   gpus=[],
											   batch_size=batch_size)

########## Benchmarking ##########

class DogCatBinaryDriver(DriverBase):

	def __init__(self, vgg_feats_model_path, vgg_svm_model_path, inception_feats_model_path, gbm_model_path, vgg_gpu_num, inception_gpu_num):
		DriverBase.__init__(self)

		self.vgg_feats_model = vgg_feats_model.VggFeaturizationModel(vgg_feats_model_path, gpu_num=vgg_gpu_num)
		self.vgg_svm_model = vgg_svm_model.VggSVM(vgg_svm_model_path)
		self.inception_feats_model = inception_feats_model.InceptionFeaturizationModel(inception_feats_model_path, gpu_num=inception_gpu_num)
		self.gbm_model = lgbm_model.ImagesGBM(gbm_model_path)

		self.thread_pool = ThreadPoolExecutor(max_workers=2)

	def _run(self, inputs, log=False):
		"""
		Parameters
		----------
		inputs : list
		   	A list of numpy float32 arrays of dimension >= (299, 299, 3)
		"""
		t1 = datetime.now()

		input_imgs = [Image.fromarray(raw_input_img.astype(np.uint8)) for raw_input_img in inputs]
		vgg_inputs = self._get_vgg_inputs(input_imgs)
		inception_inputs = input_imgs

		t2 = datetime.now()

		vgg_future = self.thread_pool.submit(lambda inputs : self._classify_vgg(self._featurize_vgg(inputs, log=log), log=log), vgg_inputs)
		inception_gbm_future = self.thread_pool.submit(lambda inputs : self._classify_gbm(self._featurize_inception(inputs, log=log), log=log), inception_inputs)

		vgg_classes = vgg_future.result()
		inception_gbm_classes = inception_gbm_future.result()

		for i in range(0, len(inception_gbm_classes)):
			if inception_gbm_classes[i] <= .5:
				inception_gbm_classes[i] = 0
			else:
				inception_gbm_classes[i] = 1

		t3 = datetime.now()

		self._log("Input processing latency: {} ms, Prediction latency: {} ms".format((t2 - t1).total_seconds() * 1000, (t3 - t2).total_seconds() * 1000), allow=log)

		return [vgg_classes[i] if vgg_classes[i] == inception_gbm_classes[i] else -1 for i in range(0, len(vgg_classes))]

	def _get_vgg_inputs(self, input_imgs):
		vgg_inputs = []
		for input_img in input_imgs:
			vgg_img = input_img.resize((224, 224)).convert('RGB')
			vgg_input = np.array(vgg_img, dtype=np.float32)
			vgg_inputs.append(vgg_input)
		return vgg_inputs

	def _benchmark_model_step(self, fn, inputs):
		begin = datetime.now()
		outputs = fn(inputs)
		end = datetime.now()
		latency_seconds = (end - begin).total_seconds()
		throughput = float(len(inputs)) / latency_seconds
		return (latency_seconds, throughput, outputs)

	def _featurize_vgg(self, inputs, log=False):
		latency, throughput, outputs = self._benchmark_model_step(self.vgg_feats_model.predict, inputs)
		return outputs

	def _featurize_inception(self, inputs, log=False):
		latency, throughput, outputs = self._benchmark_model_step(self.inception_feats_model.predict, inputs)
		return outputs

	def _classify_vgg(self, inputs, log=False):
		latency, throughput, outputs = self._benchmark_model_step(self.vgg_svm_model.predict, inputs)
		return outputs		

	def _classify_gbm(self, inputs, log=False):
		latency, throughput, outputs = self._benchmark_model_step(self.gbm_model.predict, inputs)
		return outputs

class DriverBenchmarker:
	def __init__(self, driver):
		self.driver = driver

	def benchmark(self, batch_size=1, avg_after=5, log_intermediate=False, **kwargs):
		inputs_gen_fn = lambda num_inputs : [np.random.rand(299, 299, 3) * 255 for i in range(0, num_inputs)]
		benchmarking.benchmark_function(lambda inputs : self._run(inputs, log=log_intermediate), inputs_gen_fn, batch_size, avg_after)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Image Driver 1')
    parser.add_argument('-d', '--duration', type=int, default=120, help='The maximum duration of the benchmarking process in seconds, per iteration')
    parser.add_argument('-b', '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the driver. Each configuration will be benchmarked separately.")
    parser.add_argument('-c', '--cpus', type=int, nargs='+', help="The set of cpu cores on which to run the single process driver")
    parser.add_argument('-v', '--vgg_gpu', type=int, default=0, help="The GPU on which to run the VGG featurization model")
    parser.add_argument('-i', '--inception_gpu', type=int, default=0, help="The GPU on which to run the inception featurization model")
    
    args = parser.parse_args()

    if not args.cpus:
    	raise Exception("The set of allocated cpus must be specified via the '--cpus' flag!")

    default_batch_size_confs = [2]

   	benchmarer = Driv

    for batch_size in batch_size_confs:
    	configs = get_heavy_node_configs(batch_size=batch_size,
    									 allocated_cpus=args.cpus
    									 vgg_gpus=[args.vgg_gpu],
    									 inception_gpus=[args.inception_gpu])



                benchmarker = ModelBenchmarker(model_config, queue)

                processes = []
                all_stats = []
                for _ in range(args.num_clients):
                    p = Process(target=benchmarker.run, args=(args.duration,))
                    p.start()
                    processes.append(p)
                for p in processes:
                    all_stats.append(queue.get())
                    p.join()

                cl = ClipperConnection(DockerContainerManager(redis_port=6380))
                cl.connect()
                driver_utils.save_results([model_config], cl, all_stats, "gpu_and_batch_size_experiments")
