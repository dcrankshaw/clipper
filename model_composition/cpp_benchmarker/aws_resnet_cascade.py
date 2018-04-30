import subprocess32 as subprocess
import os
import sys
import numpy as np
import time
import logging
import json
from clipper_admin import ClipperConnection, AWSContainerManager
from datetime import datetime
from containerized_utils import driver_utils
import hashlib
from itertools import combinations

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "TIMEOUT"
# CLIPPER_ADDRESS = "localhost"

RES50 = "res50"
RES152 = "res152"
ALEXNET = "alexnet"


GPUS_PER_MACHINE = 4

# REMOTE_ADDR = "172.10.0.90"
ALL_REMOTE_ADDRS = []


def get_heavy_node_config(model_name,
                          batch_size,
                          num_replicas,
                          cpus_per_replica,
                          allocated_cpus,
                          allocated_gpus,
                          remote_addr=None,
                          input_size=None):

    if model_name == ALEXNET:
        image = "gcr.io/clipper-model-comp/pytorch-alexnet:bench"
        return driver_utils.HeavyNodeConfig(name="alexnet",
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr)

    elif model_name == RES50:
        image = "gcr.io/clipper-model-comp/pytorch-res50:bench"
        return driver_utils.HeavyNodeConfig(name="res50",
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr
                                            )

    elif model_name == RES152:
        image = "gcr.io/clipper-model-comp/pytorch-res152:bench"
        return driver_utils.HeavyNodeConfig(name="res152",
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr
                                            )



def setup_clipper(addr_config_map):
    for addr, configs in addr_config_map.iteritems():
        cl = ClipperConnection(AWSContainerManager(host=addr, redis_port=6380))
        # cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        cl.connect()
        cl.stop_all(remote_addrs=ALL_REMOTE_ADDRS)
        cl.start_clipper(
            query_frontend_image="clipper/zmq_frontend:develop",
            redis_cpu_str="0",
            mgmt_cpu_str="0",
            query_cpu_str="0,16,1,17,2,18,3,19")
        time.sleep(10)
        for c in configs:
            driver_utils.setup_heavy_node(cl, c, None, DEFAULT_OUTPUT)
    time.sleep(10)
    logger.info("Clipper is set up!")


def get_clipper_batch_sizes(metrics_json):
    hists = metrics_json["histograms"]
    mean_batch_sizes = {}
    for h in hists:
        if "batch_size" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[1]
            mean = float(h[name]["mean"])
            mean_batch_sizes[model] = round(float(mean), 2)
    return mean_batch_sizes


def get_clipper_latencies(metrics_json):
    hists = metrics_json["histograms"]
    p99_latencies = {}
    mean_latencies = {}
    for h in hists:
        if "prediction_latency" in h.keys()[0] and "model" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[1]
            p99 = float(h[name]["p99"]) / 1000.0 / 1000.0
            mean = float(h[name]["mean"]) / 1000.0 / 1000.0
            p99_latencies[model] = p99
            mean_latencies[model] = mean
    return (mean_latencies, p99_latencies)


def get_clipper_queue_sizes(metrics_json):
    hists = metrics_json["histograms"]
    mean_queue_sizes = {}
    for h in hists:
        if "queue_size" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[0]
            mean = float(h[name]["mean"])
            mean_queue_sizes[model] = round(float(mean), 2)
    return mean_queue_sizes


def get_clipper_thruputs(metrics_json):
    meters = metrics_json["meters"]
    thrus = {}
    for m in meters:
        if "prediction_throughput" in m.keys()[0]:
            name = m.keys()[0]
            rate = float(m[name]["rate"])
            thrus[name] = round(float(rate), 5)
    return thrus


def get_clipper_counts(metrics_json):
    counters = metrics_json["counters"]
    counts = {}
    for c in counters:
        if "internal" not in c.keys()[0]:
            name = c.keys()[0]
            count = c[name]["count"]
            counts[name] = int(count)
    return counts


def get_profiler_stats(cur_client_metrics):
    all_latencies = []
    all_thrus = []
    all_counts = []
    all_ingest_rates = []
    for metrics_json in cur_client_metrics:
        latencies = {}
        thrus = {}
        counts = {}
        ingest_rates = {}

        data_lists = metrics_json["data_lists"]
        for l in data_lists:
            if "prediction_latencies" in l.keys()[0]:
                name = l.keys()[0]
                model = name.split(":")[0]
                cur_lats = [float(list(i.values())[0]) for i in l[name]["items"]]
                # convert to seconds
                cur_lats = np.array(cur_lats) / 1000.0 / 1000.0
                latencies[model] = cur_lats

        meters = metrics_json["meters"]
        for m in meters:
            if "prediction_throughput" in m.keys()[0]:
                name = m.keys()[0]
                model = name.split(":")[0]
                rate = float(m[name]["rate"])
                thrus[model] = round(float(rate), 5)

        for m in meters:
            if "ingest_rate" in m.keys()[0]:
                name = m.keys()[0]
                model = name.split(":")[0]
                rate = float(m[name]["rate"])
                ingest_rates[model] = round(float(rate), 5)

        counters = metrics_json["counters"]
        counts = {}
        for c in counters:
            if "num_predictions" in c.keys()[0]:
                name = c.keys()[0]
                model = name.split(":")[0]
                count = c[name]["count"]
                counts[model] = int(count)
        all_latencies.append(latencies)
        all_thrus.append(thrus)
        all_ingest_rates.append(ingest_rates)
        all_counts.append(counts)
    agg_p99_latency = {}
    agg_mean_latency = {}
    for model in all_latencies[0]:
        lats = np.hstack([c[model] for c in all_latencies]).flatten()
        agg_p99_latency[model] = round(np.percentile(lats, 99), 3)
        agg_mean_latency[model] = round(np.mean(lats), 3)

    agg_thrus = {}
    for model in all_thrus[0]:
        total_thru = np.sum([c[model] for c in all_thrus])
        agg_thrus[model] = total_thru

    agg_ingest_rates = {}
    for model in all_ingest_rates[0]:
        total_ingest_rate = np.sum([c[model] for c in all_ingest_rates])
        agg_ingest_rates[model] = total_ingest_rate

    agg_counts = {}
    for model in all_counts[0]:
        total_count = np.sum([c[model] for c in all_counts])
        agg_counts[model] = total_count

    return (agg_mean_latency, agg_p99_latency, agg_thrus, agg_counts, agg_ingest_rates)


def load_clipper_metrics(clipper_path):
    with open(clipper_path, "r") as clipper_file:
        clipper_metrics_str = clipper_file.read().strip()
        if clipper_metrics_str[-1] == ",":
            clipper_metrics_str = clipper_metrics_str.rstrip(",")
            clipper_metrics_str += "]"
        try:
            clipper_metrics = json.loads(clipper_metrics_str)
        except ValueError:
            # logger.warn("Unable to parse clipper metrics: {}. Skipping...".format(e))
            return None
    return clipper_metrics


def print_clipper_metrics(clipper_metrics):
    results_dict = {}
    results_dict["batch_sizes"] = get_clipper_batch_sizes(clipper_metrics)
    results_dict["queue_sizes"] = get_clipper_queue_sizes(clipper_metrics)
    results_dict["clipper_thrus"] = get_clipper_thruputs(clipper_metrics)
    results_dict["clipper_counts"] = get_clipper_counts(clipper_metrics)
    results_dict["mean_lats"], results_dict["p99_lats"] = get_clipper_latencies(clipper_metrics)
    logger.info(("\nqueue sizes: {queue_sizes}"
                 "\nbatch sizes: {batch_sizes}"
                 "\nclipper p99 lats: {p99_lats}"
                 "\nclipper mean lats: {mean_lats}\n").format(**results_dict))


def print_stats(cur_client_metrics, trial_num):
    cur_client_metrics = [c[trial_num] for c in cur_client_metrics]
    results_dict = {}
    results_dict["client_mean_lats"], results_dict["client_p99_lats"], \
        results_dict["client_thrus"], results_dict["client_counts"], \
        results_dict["ingest_rates"] = get_profiler_stats(cur_client_metrics)
    logger.info(("\nThroughput: {client_thrus}\nP99 lat: {client_p99_lats}"
                "\nMean lat: {client_mean_lats}"
                 "\nIngest rates: {ingest_rates}\n").format(**results_dict))
    return results_dict


# def load_metrics(client_path, clipper_path):
def load_metrics(client_path, clipper_path=None):
    with open(client_path, "r") as client_file:

        client_metrics_str = client_file.read().strip()
        if client_metrics_str[-1] == ",":
            client_metrics_str = client_metrics_str.rstrip(",")
            client_metrics_str += "]"
        try:
            client_metrics = json.loads(client_metrics_str)
        except ValueError:
            # logger.warn("Unable to parse client metrics: {}. Skipping...".format(e))
            return None
    if clipper_path is not None:
        with open(clipper_path, "r") as clipper_file:
            clipper_metrics_str = clipper_file.read().strip()
            if clipper_metrics_str[-1] == ",":
                clipper_metrics_str = clipper_metrics_str.rstrip(",")
                clipper_metrics_str += "]"
            try:
                clipper_metrics = json.loads(clipper_metrics_str)
            except ValueError:
                # logger.warn("Unable to parse clipper metrics: {}. Skipping...".format(e))
                return None
        return client_metrics, clipper_metrics
    else:
        return client_metrics


def load_lineage(lineage_path):
    with open(lineage_path, "r") as f:
        parsed = [json.loads(l) for l in f]
    return parsed


def get_arrival_proc_file(lam, cv):
    if cv == 1:
        arrival_file_name = "{lam}.deltas".format(lam=lam)
    else:
        arrival_file_name = "{lam}_{cv}.deltas".format(lam=lam, cv=cv)

    arrival_delay_file = os.path.join(("/home/ubuntu/plots-model-comp-paper/experiments/"
                                       "cached_arrival_processes/{f}").format(f=arrival_file_name))
    return arrival_delay_file


def get_arrival_proc_files(lam, cv, num_clients, out_dir):
    arrival_delay_file = get_arrival_proc_file(lam, cv)
    if num_clients == 1:
        return [arrival_delay_file]
    # else:
    #     div = get_arrival_proc_file(int(lam/num_clients), cv)
    #     return [div for _ in range(num_clients)]

    with open(arrival_delay_file, "r") as f:
        deltas = np.array([float(l.strip()) for l in f]).flatten()
    arrival_history = np.cumsum(deltas)

    # Assign each query to a specific client
    tags = np.array([np.random.randint(num_clients) for _ in range(len(arrival_history))])
    tagged_arrival_diffs = [np.diff(arrival_history[tags == i]) for i in range(num_clients)]
    deltas_base = os.path.basename(arrival_delay_file)
    out_files = []
    for i in range(num_clients):
        fname = os.path.join(out_dir, "CLIENT-{}-{}".format(i, deltas_base))
        out_files.append(fname)
        with open(fname, "w") as f:
            for d in tagged_arrival_diffs[i]:
                f.write("{}\n".format(d))

    return out_files


def run_e2e(addr_config_map, name_addr_map, trial_length, driver_path, profiler_cores_strs,
            lam, cv, num_clients, slo):
    assert len(addr_config_map) >= 1
    setup_clipper(addr_config_map)
    # clipper_address = CLIPPER_ADDRESS
    cls = [ClipperConnection(AWSContainerManager(host=addr, redis_port=6380)) for addr in addr_config_map]
    # cls = [ClipperConnection(DockerContainerManager(redis_port=6380)) for addr in addr_config_map]
    for cl in cls:
        cl.connect()

    time.sleep(30)
    log_dir = "/tmp/resnet_cascade_logs_{ts:%y%m%d_%H%M%S}".format(ts=datetime.now())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    def run(target_throughput, num_trials, name, arrival_process):
        for cl in cls:
            cl.drain_queues()
        time.sleep(10)
        for cl in cls:
            cl.drain_queues()
        time.sleep(10)
        arrival_delay_files = get_arrival_proc_files(lam, cv, num_clients, log_dir)
        try:
            procs = {}
            for client_num in range(num_clients):
                log_path = os.path.join(log_dir, "{n}-{t}-{p}-{c}".format(n=name,
                                                                    t=target_throughput,
                                                                    p=arrival_process,
                                                                    c=client_num))
                cmd = ["numactl", "-C", profiler_cores_strs[client_num],
                    os.path.abspath(driver_path),
                    "--target_throughput={}".format(target_throughput),
                    "--request_distribution={}".format(arrival_process),
                    "--trial_length={}".format(trial_length),
                    "--num_trials={}".format(num_trials),
                    "--log_file={}".format(log_path),
                    "--clipper_address_alexnet={}".format(name_addr_map[ALEXNET]),
                    "--clipper_address_res50={}".format(name_addr_map[RES50]),
                    "--clipper_address_res152={}".format(name_addr_map[RES152]),
                    "--request_delay_file={}".format(arrival_delay_files[client_num])]
                if client_num == 0:
                    cmd.append("--get_clipper_metrics")

                logger.info("Driver command: {}".format(" ".join(cmd)))
                client_path = "{p}-client_metrics.json".format(p=log_path)
                lineage_paths = {m: "{p}-{m}-query_lineage.txt".format(m=m, p=log_path)
                                for m in [ALEXNET, RES50, RES152]}
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                procs[client_num] = (proc, log_path, client_path, lineage_paths)
            clipper_paths = ["{p}-clipper_metrics_{a}.json".format(
                p=log_path, a=a) for a in addr_config_map.keys()]

            recorded_trials = 0
            summary_results = []
            done = False
            while recorded_trials < num_trials and not done:
                time.sleep(5)
                cur_client_metrics = []
                for client_num in procs:
                    proc, log_path, client_path, lineage_paths = procs[client_num]
                    if proc.poll() is not None:
                        if proc.returncode != 0:
                            logger.warn("Driver process {client} finished with return code {code}".format(
                                client=client_num, code=proc.returncode))
                            done = True
                            break
                    if not (os.path.exists(client_path)):
                        logger.info("metrics don't exist yet")
                        continue
                    client_metrics = load_metrics(client_path)
                    if client_metrics is not None:
                        cur_client_metrics.append(client_metrics)
                print_stats_this_iter = False
                if len(cur_client_metrics) == num_clients:
                    new_recorded_trials = min([len(c) for c in cur_client_metrics])
                    if new_recorded_trials > recorded_trials:
                        recorded_trials = new_recorded_trials
                        summary_results.append(print_stats(cur_client_metrics, new_recorded_trials - 1))
                        print_stats_this_iter = True

                if print_stats_this_iter:
                    for clipper_path in clipper_paths:
                        if not (os.path.exists(clipper_path)):
                            logger.info("Clipper metrics don't exist yet")
                            continue
                        clipper_metrics = load_clipper_metrics(clipper_path)
                        if clipper_metrics and len(clipper_metrics) >= recorded_trials:
                            print_clipper_metrics(clipper_metrics[recorded_trials - 1])

            all_client_metrics = []
            for client_num in procs:
                proc, log_path, client_path, lineage_paths = procs[client_num]
                prof_stderr = proc.stderr.read().strip()
                if len(prof_stderr) > 0:
                    logger.info("stderr: {}".format(prof_stderr))
                try:
                    loaded_metrics = load_metrics(client_path)
                    # lineages = {name: load_lineage(p) for name, p in lineage_paths.items()}
                    lineages = None
                    if loaded_metrics is not None:
                        all_client_metrics.append(loaded_metrics)
                    else:
                        logger.error("Error loading final metrics")
                except ValueError as e:
                    logger.error("Unable to parse final metrics")
                    raise e
            return driver_utils.Results(all_client_metrics,
                                        None,
                                        summary_results,
                                        lineages)
        except Exception as e:
            logger.exception(e)
        finally:
            for c in procs:
                procs[c][0].terminate()

    # Warmup with throughput of 1000 seems to choke the system, so warm up with 100 qps instead
    run(100, 10, "warmup", "constant")
    throughput_results = run(0, 25, "throughput", "file")

    for cl in cls:
        cl.stop_all(remote_addrs=ALL_REMOTE_ADDRS)
    return throughput_results


class BenchmarkConfigurationException(Exception):
    pass


def hash_file(fname):
    with open(fname, "rb") as f:
        fbytes = f.read()
        return hashlib.sha256(fbytes).hexdigest()



def run_experiment_for_config(config):
    cpu_map = {
        ALEXNET_CLIPPER_ADDR: range(8, 16),
        RES50_CLIPPER_ADDR: range(8, 16),
        RES152_CLIPPER_ADDR: range(8, 16)
    }

    gpu_map = {
        ALEXNET_CLIPPER_ADDR: range(GPUS_PER_MACHINE),
        RES50_CLIPPER_ADDR: range(GPUS_PER_MACHINE),
        RES152_CLIPPER_ADDR: range(GPUS_PER_MACHINE)
    }

    # NOTE: Distribution of models among nodes assumes all models are running on V100s

    total_resource_bundle_map = {}

    combined_gpu_usage = 0
    for name, c in config["node_configs"].iteritems():
        if c["gpu_type"] in ["None", "none", None]:
            gpu_multiple = 0
        else:
            gpu_multiple = 1

        total_gpus = c["num_replicas"] * gpu_multiple
        combined_gpu_usage += total_gpus
        # Make sure that each model can fit on a machine by itself at least.
        assert total_gpus <= 4
        total_resource_bundle_map[name] = {
            "gpus": total_gpus,
            "cpus": c["num_replicas"] * c["num_cpus"]
        }

    all_models = [RES152, ALEXNET, RES50]

    # Default addr config map is each model on separate machine
    model_name_to_addr_map = {
        ALEXNET: ALEXNET_CLIPPER_ADDR,
        RES50: RES50_CLIPPER_ADDR,
        RES152: RES152_CLIPPER_ADDR
    }

    # First check if all 3 models can fit onto one machine
    if combined_gpu_usage <= GPUS_PER_MACHINE:
        # If so, assign all models to ALEXNET_CLIPPER_ADDR
        model_name_to_addr_map = dict([(m, ALEXNET_CLIPPER_ADDR) for m in all_models])
        logger.info("Running all models on {}".format(ALEXNET_CLIPPER_ADDR))
    else:
        # Now check if any 2 models can fit onto the same machine
        model_pairs = combinations(all_models, 2)
        for a, b in model_pairs:
            if total_resource_bundle_map[a]["gpus"] + total_resource_bundle_map[b]["gpus"] <= GPUS_PER_MACHINE:
                # Put model A and B onto same machine
                model_name_to_addr_map[a] = model_name_to_addr_map[b]
                logger.info("Running {a} and {b} on same physical machine: {machine}".format(
                    a=a, b=b, machine=model_name_to_addr_map[b]))
                break


    def get_cpus(num, addr):
        try:
            return [cpu_map[addr].pop() for _ in range(num)]
        except IndexError:
            msg = "Ran out of out available CPUs"
            logger.error(msg)
            raise BenchmarkConfigurationException(msg)

    def get_gpus(num, gpu_type, addr):
        if gpu_type == "none":
            return None
        else:
            if gpu_type != "v100":
                raise BenchmarkConfigurationException("Config required a k80. Please provision separately.")
            try:
                return [gpu_map[addr].pop() for _ in range(num)]
            except IndexError:
                msg = "Ran out of available GPUs"
                logger.error(msg)
                raise BenchmarkConfigurationException(msg)

    try:
        node_configs = []

        addr_config_map = {}
        # Figure out which nodes are actually going to be used
        for v in model_name_to_addr_map.values():
            addr_config_map[v] = []

        for name, c in config["node_configs"].iteritems():
            addr = model_name_to_addr_map[name]
            node = get_heavy_node_config(model_name=name,
                                batch_size=int(c["batch_size"]),
                                num_replicas=c["num_replicas"],
                                cpus_per_replica=c["num_cpus"],
                                allocated_cpus=get_cpus(c["num_cpus"]*c["num_replicas"], addr),
                                allocated_gpus=get_gpus(c["num_replicas"], c["gpu_type"], addr))
            node_configs.append(node)
            addr_config_map[addr].append(node)

    except BenchmarkConfigurationException as e:
        logger.error("Error provisioning for requested configuration. Skipping.\n"
                     "Reason: {reason}\nBad config was:\n{conf}".format(reason=e, conf=json.dumps(config, indent=2)))
        return None

    addr_config_str = ""
    for a, cs in addr_config_map.iteritems():
        addr_config_str += "{a}: {cs}\n".format(a=a, cs=json.dumps([c.__dict__ for c in cs], indent=2))


    print("\n\n\nADDR CONFIG MAP:\n{}".format(json.dumps(dict([(a, [c.__dict__ for c in cs]) for a, cs in addr_config_map.iteritems()]), indent=2)))
    print("NAME ADDR MAP:\n:{}".format(json.dumps(model_name_to_addr_map, indent=2)))

    lam = config["lam"]
    cv = config["cv"]
    slo = config["slo"]
    cost = config["cost"]
    utilization = config["utilization"]
    config["deltas_file_path"] = get_arrival_proc_file(lam, cv)
    config["deltas_file_md5sum"] = hash_file(config["deltas_file_path"])
    if "latency_percentage" not in config:
        config["latency_percentage"] = 1.0
    latency_perc = config["latency_percentage"]

    # results_dir = "pipeline_three_prof_underestimate_slo_{slo}_cv_{cv}_util_{util}".format(
    #     slo=slo, cv=cv, util=utilization)
    results_dir = "pipeline_three_e2e_sys_comp/util_{util}".format(util=utilization)
    reps_str = "_".join(["{name}-{reps}".format(name=c["name"], reps=c["num_replicas"])
                         for c in config["node_configs"].values()])
    results_fname = "aws_latency_percentage_{perc}_lambda_{lam}".format(
        lam=lam, perc=latency_perc)


    # For client on standalone machine
    client_cpu_strs = [
        # "4,20,5,21,6,22,7,23"
        "0,1,2,3,4,5,6,7,32,33,34,35,36,37,38,39",
        "16,17,18,19,20,21,22,23,48,49,50,51,52,43,54,55"
    ]

    num_clients = 2

    throughput_results = run_e2e(
        addr_config_map, model_name_to_addr_map, 2000, "../../release/src/inferline_client/resnet_cascade",
        client_cpu_strs, int(lam / num_clients), cv, num_clients, slo)
    driver_utils.save_results_cpp_client(
        dict([(a, [c.__dict__ for c in cs]) for a, cs in addr_config_map.iteritems()]),
        throughput_results,
        None,
        results_dir,
        prefix=results_fname,
        used_config=config)


if __name__ == "__main__":
    global ALEXNET_CLIPPER_ADDR
    global RES50_CLIPPER_ADDR
    global RES152_CLIPPER_ADDR


    base_path = os.path.expanduser("~/plots-model-comp-paper/experiments/e2e_sys_comp_pipeline_three/util_1.0")

    config_paths = [
        # "aws_resnet_cascade_ifl_configs_slo_1.0_cv_0.1_higher_cost.json",
        "aws_resnet_cascade_ifl_configs_slo_1.0_cv_1.0_higher_cost.json",
        "aws_resnet_cascade_ifl_configs_slo_1.0_cv_4.0_higher_cost.json",
        "aws_resnet_cascade_ifl_configs_slo_0.5_cv_0.1_higher_cost.json"
    ]


    config_paths = [os.path.join(base_path, c) for c in config_paths]

    # for config_path in args.config_paths:
    for config_path in config_paths:
        print(config_path)
        with open(os.path.abspath(os.path.expanduser(config_path)), "r") as f:
            provided_configs = json.load(f)

        for config in provided_configs:
            ALEXNET_CLIPPER_ADDR = os.environ["ALEXNET_CLIPPER_ADDR"]
            RES50_CLIPPER_ADDR = os.environ["RES50_CLIPPER_ADDR"]
            # RES152_CLIPPER_ADDR = os.environ["RES152_CLIPPER_ADDR"]
            RES152_CLIPPER_ADDR = os.environ["RES50_CLIPPER_ADDR"]
            run_experiment_for_config(config)
    sys.exit(0)
