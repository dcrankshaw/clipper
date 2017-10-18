import json

class HeavyNodeConfig(object):
    def __init__(self,
                 model_name,
                 input_type,
                 allocated_cpus,
                 cpus_per_replica,
                 num_replicas=1,
                 gpus=[],
                 batch_size=1):
        self.model_name = model_name
        self.input_type = input_type
        self.allocated_cpus = allocated_cpus
        self.cpus_per_replica = cpus_per_replica
        self.num_replicas = num_replicas
        self.gpus = gpus
        self.batch_size = batch_size

    def to_json(self):
        return json.dumps(self.__dict__)

def save_results(configs, client_metrics, results_dir):
    """
    Parameters
    ----------
    configs : list(HeavyNodeConfig)
        The configs for any models deployed


    """

    results_dir = os.path.abspath(os.path.expanduser(results_dir))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info("Created experiments directory: %s" % results_dir)

    results_obj = {
        "node_configs": [c.__dict__ for c in configs],
        "client_metrics": client_metrics,
    }
    results_file = os.path.join(results_dir,
                                "results-{:%y%m%d_%H%M%S}.json".format(datetime.datetime.now()))
    with open(results_file, "w") as f:
        json.dump(results_obj, f, indent=4)
        logger.info("Saved results to {}".format(results_file))