### Setup ###

* In the [DEBUGGING](DEBUGGING) subdirectory, you will find a set of experimental configurations (files ending in .json)
specifying a few **SPD** control parameters: replication factor and batch size, as well as 
the set of CPU and GPU affinities for each replica (if 2 replicas are used, the first and second
elements of the "cpu_affinities" and "gpu_affinities" lists correspond to the resource allocations
for the first and second replicas, respectively). Other control parameters for the **experiment**
include: the number of trias, the trial length, and a path to the arrival process to be used.
Note that the SLO is only included for result file logging purposes; it has no impact on the 
experiment.

* There are also two tagged arrival processes. These processes are files with lines of the form:
    
    <request_delay_in_milliseconds>,<replica_number>

The nth replica will only processes requests on lines corresponding to replica n.
For debugging purposes, there are two processes: "363_tagged_1.deltas" that is tagged
for a single replica (all lines end in replica number "0") and "363_tagged_2.deltas"
that is tagged for two replicas (lines end with either "0" or "1").

* To execute an experiment with a configuration, run the following:

  ```
  $ python run_distributed_driver.py <PATH_TO_EXPERIMENTAL_CONFIG>
  ```

### Reproducing Problematic Behavior ###

The following steps can be used to reproduce the discussed, problematic behavior:

1. Run the experimental configuration located at [DEBUGGING/363_1_rep_config_A.json](DEBUGGING/363_1_rep_config_A.json).
   This will launch a single replica and benchmark it using an arrival process with a lambda
   of 363. The replica will be launched on GPUs 0 and 1 and **physical** cores 0-3. It will use
   a batch size of 80.

   * As the experiment runs, pay attention to the "p99_batch_predict" latencies that are
     emitted. These will be in the range of 300-400 milliseconds, printed in second resolution (.3 - .4).
     These are latencies for the **critical path** calls to `resnet.predict` and `inception.predict`. You can
     see how they are calculated here: https://github.com/Corey-Zumar/clipper-1/blob/d8bb75ca5c76f06818002da49673ee6de8e09e6a/model_composition/image_driver_1/single_process/driver.py#L225
     

2. Run the experimental configuration located at [DEBUGGING/363_2_rep_config.json]("DEBUGGING/363_2_rep_config.json")
   This will launch two replicas and benchmark them using an arrival process with a lambda of 363
   that is tagged for two replicas. The first replica will be launched on GPUs 0 and 1 and **physical**
   cores 0-3. The second replica will be launched on GPUs 2 and 3 and **physical** cores 4-7. Both
   replicas will use a batch size of 80.

   * As the experiment runs, note that, despite the fact that we are running separate python processes on 
     separate GPUs and physical CPUs, "p99_batch_predict" latencies increase dramatically (on the order of 600 milliseconds)
     for each replica.
     
3. Simultaneously run both of the following experimental configurations: [DEBUGGING/363_1_rep_config_A.json](DEBUGGING/363_1_rep_config_A.json) and [DEBUGGING/363_1_rep_config_B.json](DEBUGGING/363_1_rep_config_B.json). Each experiment will launch one replica and use the arrival process from step (1). The first experiment is run on GPUs 0 and 1 and **physical** cores 0-3. The second experiment is run on GPUs 2 and 3 and **physical** cores 4-7.

    * As the experiment runs, note that "p99_batch_predict" latencies are consistent with the latencies seen in **step (1)** (300 - 400     milliseconds). We do not observe an increase in critical path latency like we do in step (2). 
