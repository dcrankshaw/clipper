# Image Driver 1

## Background

This driver makes use of 4 heavyweight models: A VGG model for image featurization, an Inception V3 model for image featurization, 
an SVM with Kernel PCA for feature classification, and a light boosted gradient model (LGBM) for feature classification. The driver file,
[driver.py](driver.py), will help you benchmark each of this models with Clipper in isolation (one model at a time). The remaining sections
of this README will get you started with the benchmarking process.

## Activate your Clipper Anaconda environment
Before proceeding, make sure to activate your Anaconda environment if it is not already activated. This can be done by running:
```sh
$ source activate clipper
```

## Pre-requisite: Setting up infrastructure
If the [high-perf-clipper branch](https://github.com/dcrankshaw/clipper/tree/high-perf-clipper) has been updated since you last
performed benchmarking, follow the subsequent instructions to build or install the latest benchmarking infrastructure components.

### Install the latest versions of Clipper's python modules:
Run the following command from the [model_composition directory](../../.)
```sh
$ ./setup_env.sh
```

### Build the latest Clipper docker images 
Run the following command from the clipper root directory to build the latest Clipper docker images 
(ZMQ Frontend, Management Frontend, etc):

```sh
$ ./bin/build_docker_images.sh
```

### Build the latest model docker images
Run the following command from the [containerized image driver 1 directory](.) 
to build docker images for models used by Image Driver 1:

```sh
$ ./containers/build_docker_images.sh
```

## Benchmarking with [driver.py](driver.py)

### The Driver API

The driver accepts the following arguments:
- **model_name**: The name of the model to benchmark. Must be one of the following: `vgg`, `inception`, `svm`, `lgbm`
  * This argument is REQUIRED
  
- **duration**: The duration for which each iteration of the benchmark should run, in seconds
  * If unspecified, this argument has a default value of 120 seconds
  
- **batch_sizes**: The batch size configurations to benchmark. Each configuration will be benchmarked separately.
  * If unspecified, the driver will benchmark a single batch size configuration of size `2`.
  
- **num_replicas**: The "number of replicas" configurations to benchmark. Each configuration will be benchmarked separately.
  * If unspecified, the driver will benchmark a single "number of replicas" configuration of size `1`

### Example
As an example, consider the following driver command:

```sh
$ python driver.py --duration_seconds 120 --model_name vgg --num_replicas 1 2 3 4 --batch_sizes 1 2 4 8 16 32
```

This command specifies `4` different replica configurations and `6` batch size configurations. Therefore, a total of
`4 * 6 = 24` benchmarking iterations will occur (one for each combination of configurations). Each iteration will last a maximum
of 120 seconds and will use the VGG model.

### Avoiding CPU resource conflicts


## Monitoring the benchmarking process
Once you've started a benchmark, there are some useful tools and logs that you can use to monitor behavior.

### Monitoring CPU usage
If you're running a CPU-intensive model on a set of cores, `{c_1, ..., c_n}`, you can use [htop](http://hisham.hm/htop/)
to monitor for higher activity on those cores. 

### Monitoring GPU usage
If you're benchmarking the VGG or Inception models on a GPU, you can make sure that the model is using the GPU via the 
[nvidia-smi](http://developer.download.nvidia.com/compute/cuda/6_0/rel/gdk/nvidia-smi.331.38.pdf) command.

If no tasks are running on the GPU, the output of `nvidia-smi` will look similar to the following:

![Image of Unused Nvidia-Smi](nvidia-smi-unused.jpg)

In contrast, if tasks are running, you should see non-zero (hopefully high) memory and utilization, as well as one or more active processes (Note: This output was obtained from a machine with 2 GPUs):

![Image of Used Nvidia-Smi](nvidia-smi-used.jpg)


