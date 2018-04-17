from __future__ import absolute_import, division, print_function
# import docker
import logging
import os
import random
import requests
from ..container_manager import (
    create_model_container_label, parse_model_container_label,
    ContainerManager, CLIPPER_DOCKER_LABEL, CLIPPER_MODEL_CONTAINER_LABEL,
    CLIPPER_QUERY_FRONTEND_CONTAINER_LABEL,
    CLIPPER_MGMT_FRONTEND_CONTAINER_LABEL, CLIPPER_INTERNAL_RPC_PORT,
    CLIPPER_INTERNAL_MANAGEMENT_PORT)
from ..exceptions import ClipperException
# import subprocess32 as subprocess
from fabric.api import run, env, shell_env, warn_only, local, show, hide, output as fab_output
# from fabric.context_managers import shell_env

env.key_filename = os.path.expanduser("~/.ssh/aws_rsa")


logger = logging.getLogger(__name__)


class AWSContainerManager(ContainerManager):
    def __init__(self,
                 host="localhost",
                 key_path=os.path.expanduser("~/.ssh/aws_rsa"),
                 clipper_query_port=1337,
                 clipper_management_port=1338,
                 clipper_rpc_port=7000,
                 redis_ip=None,
                 redis_port=6379,
                 docker_network="clipper_network",
                 extra_container_kwargs={}):

        """
        Parameters
        ----------
        docker_ip_address : str, optional
            The public hostname or IP address at which the Clipper Docker
            containers can be accessed via their exposed ports. This should almost always
            be "localhost". Only change if you know what you're doing!
        clipper_query_port : int, optional
            The port on which the query frontend should listen for incoming prediction requests.
        clipper_management_port : int, optional
            The port on which the management frontend should expose the management REST API.
        clipper_rpc_port : int, optional
            The port to start the Clipper RPC service on.
        redis_ip : str, optional
            The address of a running Redis cluster. If set to None, Clipper will start
            a Redis container for you.
        redis_port : int, optional
            The Redis port. If ``redis_ip`` is set to None, Clipper will start Redis on this port.
            If ``redis_ip`` is provided, Clipper will connect to Redis on this port.
        docker_network : str, optional
            The docker network to attach the containers to. You can read more about Docker
            networking in the
            `Docker User Guide <https://docs.docker.com/engine/userguide/networking/>`_.
        extra_container_kwargs : dict
            Any additional keyword arguments to pass to the call to
            :py:meth:`docker.client.containers.run`.
        """
        self.host = host
        env.host_string = self.host
        env.key_filename = key_path
        env.colorize_errors = True
        env.disable_known_hosts = True
        fab_output["running"] = True
        fab_output["debug"] = True






        self.public_hostname = self.host
        self.clipper_query_port = clipper_query_port
        self.clipper_management_port = clipper_management_port
        self.clipper_rpc_port = clipper_rpc_port
        self.redis_ip = redis_ip
        if redis_ip is None:
            self.external_redis = False
        else:
            self.external_redis = True
        self.redis_port = redis_port
        if docker_network is "host":
            raise ClipperException(
                "AWSContainerManager does not support running Clipper on the "
                "\"host\" docker network. Please pick a different network name")
        self.docker_network = docker_network

        self.extra_container_kwargs = extra_container_kwargs.copy()

        # Merge Clipper-specific labels with any user-provided labels
        if "labels" in self.extra_container_kwargs:
            self.common_labels = self.extra_container_kwargs.pop("labels")
            self.common_labels.update({CLIPPER_DOCKER_LABEL: ""})
        else:
            self.common_labels = {CLIPPER_DOCKER_LABEL: ""}

        # self.docker_client = docker.from_env()
        # container_args = {
        #     "network": self.docker_network,
        #     "detach": True,
        # }
        #
        # self.extra_container_kwargs.update(container_args)

    def _host_is_local(self):
        return self.host == "localhost"

    def _execute(self, *args, **kwargs):
        if self._host_is_local():
            self._execute_local(*args, **kwargs)
        else:
            with show("everything"):
                run(*args, **kwargs)

    def _execute_local(self, *args, **kwargs):
        # local is not currently capable of simultaneously printing and
        # capturing output, as run/sudo do. The capture kwarg allows you to
        # switch between printing and capturing as necessary, and defaults to
        # False. In this case, we need to capture the output and return it.
        if "capture" not in kwargs:
            kwargs["capture"] = True
        # fabric.local() does not accept the "warn_only"
        # key word argument, so we must remove it before
        # calling
        print(fab_output)
        if "warn_only" in kwargs:
            del kwargs["warn_only"]
            # Forces execution to continue in the face of an error,
            # just like warn_only=True
            with warn_only():
                with show("everything"):
                    result = local(*args, **kwargs)
        else:
            with show("everything"):
                result = local(*args, **kwargs)
        return result

    def start_clipper(self, query_frontend_image, mgmt_frontend_image,
                      cache_size, redis_cpu_str="0", mgmt_cpu_str="1", query_cpu_str="2-11"):
        """
        Parameters
        ----------
        redis_cpu_str : str
            The VIRTUAL CPU(s) to which to assign Clipper's Redis store
        mgmt_cpu_str : str
            The VIRTUAL CPU(s) to which to assign Clipper's management frontend
        query_cpu_str : str
            The VIRTUAL CPU(s) to which to assign Clipper's query frontend
        """

        # with hide("output", "warnings", "running"):
        self._execute("docker network create {}".format(self.docker_network),
                warn_only=True)

        if not self.external_redis:
            redis_labels_str = " ".join(["-l {k}=\"{v}\"".format(k=k, v=v) for k, v in self.common_labels.iteritems()])
            logger.info("Starting managed Redis instance in Docker")
            redis_name = "redis-{}".format(random.randint(0, 100000))
            redis_docker_cmd = ("docker run -d --network {nw} {labels} "
                " --cpuset-cpus=\"{cpus}\" "
                            "--name {name} {ports} redis:alpine "
                            "redis-server --port {redis_port}").format(
                                nw=self.docker_network,
                                labels=redis_labels_str,
                                cpus=redis_cpu_str,
                                name=redis_name,
                                ports="-p {r}:{r}".format(r=self.redis_port),
                                redis_port=self.redis_port)
            logger.info(redis_docker_cmd)
            self._execute(redis_docker_cmd)
            self.redis_ip = redis_name

        mgmt_labels = self.common_labels.copy()
        mgmt_labels[CLIPPER_MGMT_FRONTEND_CONTAINER_LABEL] = ""
        mgmt_labels_str = " ".join(["-l {k}={v}".format(k=k, v=v) for k, v in mgmt_labels.iteritems()])
        mgmt_name = "mgmt_frontend-{}".format(random.randint(0, 100000))
        mgmt_cmd = "--redis_ip={redis_ip} --redis_port={redis_port}".format(
            redis_ip=self.redis_ip, redis_port=self.redis_port)
        mgmt_docker_cmd = ("docker run -d --network {nw} {labels} "
                    "--cpuset-cpus=\"{cpus}\" --name {name} {ports} "
                    "{image} {cmd}").format(
                        nw=self.docker_network,
                        labels=mgmt_labels_str,
                        cpus=mgmt_cpu_str,
                        name=mgmt_name,
                        ports="-p {hostport}:{containerport}".format(
                            hostport=self.clipper_management_port,
                            containerport=CLIPPER_INTERNAL_MANAGEMENT_PORT),
                        image=mgmt_frontend_image,
                        cmd=mgmt_cmd)
        logger.info(mgmt_docker_cmd)
        self._execute(mgmt_docker_cmd)




        # mgmt_labels = self.common_labels.copy()
        # mgmt_labels[CLIPPER_MGMT_FRONTEND_CONTAINER_LABEL] = ""
        # self.docker_client.containers.run(
        #     mgmt_frontend_image,
        #     mgmt_cmd,
        #     name="mgmt_frontend-{}".format(
        #         random.randint(0, 100000)),  # generate a random name
        #     ports={
        #         '%s/tcp' % CLIPPER_INTERNAL_MANAGEMENT_PORT:
        #         self.clipper_management_port
        #     },
        #     labels=mgmt_labels,
        #     cpuset_cpus=mgmt_cpu_str,
        #     **self.extra_container_kwargs)


        query_cmd = "--redis_ip={redis_ip} --redis_port={redis_port}".format(
            redis_ip=self.redis_ip, redis_port=self.redis_port)
        # query_ports = [4455, 4456, 9999, 1337, 7010, 7011, CLIPPER_INTERNAL_RPC_PORT]
        query_ports = [4455, 4456, 1337, 7010, 7011, CLIPPER_INTERNAL_RPC_PORT]
        query_ports_str = " ".join(["-p {p}:{p}".format(p=p) for p in query_ports])
        query_labels = self.common_labels.copy()
        query_labels[CLIPPER_QUERY_FRONTEND_CONTAINER_LABEL] = ""
        query_labels_str = " ".join(["-l {k}={v}".format(k=k, v=v) for k, v in query_labels.iteritems()])
        query_name="query_frontend-{}".format(random.randint(0, 100000))

        # NOTE(crankshaw): I have no idea why this matters, but it's critical to run
        # the query frontend with the --privileged flag and the sys_ptrace capability
        # to get good performance. Running without those flags significantly reduces
        # performance.
        query_docker_cmd = ("docker run -d --privileged --cap-add=SYS_PTRACE "
                            "--network {nw} {labels} "
                            "--cpuset-cpus=\"{cpus}\" --name {name} {ports} "
                    "{image} {cmd}").format(
                        nw=self.docker_network,
                        labels=query_labels_str,
                        cpus=query_cpu_str,
                        name=query_name,
                        ports=query_ports_str,
                        image=query_frontend_image,
                        cmd=query_cmd)
        logger.info(query_docker_cmd)

        # subprocess.check_call(query_docker_cmd.split())

        self._execute(query_docker_cmd)
        self.query_frontend_name = query_name


        # query_cmd = "--redis_ip={redis_ip} --redis_port={redis_port}".format(
        #     redis_ip=self.redis_ip,
        #     redis_port=self.redis_port,
        #     cache_size=cache_size)
        # query_labels = self.common_labels.copy()
        # query_labels[CLIPPER_QUERY_FRONTEND_CONTAINER_LABEL] = ""
        # self.docker_client.containers.run(
        #     query_frontend_image,
        #     query_cmd,
        #     name="query_frontend-{}".format(
        #         random.randint(0, 100000)),  # generate a random name
        #     ports={
        #         '4455/tcp': 4455,
        #         '4456/tcp': 4456,
        #         '9999/tcp': 9999,  # for gdbserver
        #         '1337/tcp': 1337,
        #         '7010/tcp': 7010,
        #         '7011/tcp': 7011,
        #         '%s/tcp' % CLIPPER_INTERNAL_RPC_PORT: self.clipper_rpc_port
        #     },
        #     labels=query_labels,
        #     cpuset_cpus=query_cpu_str,
        #     cap_add=["sys_ptrace"],
        #     privileged=True,
        #     **self.extra_container_kwargs)
        self.connect()

    def connect(self):
        # No extra connection steps to take on connection
        return

    def deploy_model(self, name, version, input_type, image, num_replicas=1, **kwargs):
        # Parameters
        # ----------
        # image : str
        #     The fully specified Docker imagesitory to deploy. If using a custom
        #     registry, the registry name must be prepended to the image. For example,
        #     "localhost:5000/my_model_name:my_model_version" or
        #     "quay.io/my_namespace/my_model_name:my_model_version"
        self.set_num_replicas(name, version, input_type, image, num_replicas, **kwargs)

    def deploy_model_remote(self, name, version, input_type, image, remote_addr, num_replicas=1,
                            **kwargs):
        self.set_num_replicas_remote(name, version, input_type, image, remote_addr, num_replicas,
                                     **kwargs)

    # def _get_replicas(self, name, version):
    #     containers = self.docker_client.containers.list(
    #         filters={
    #             "label":
    #             "{key}={val}".format(
    #                 key=CLIPPER_MODEL_CONTAINER_LABEL,
    #                 val=create_model_container_label(name, version))
    #         })
    #     return containers

    def get_num_replicas(self, name, version):
        return -1
        # return len(self._get_replicas(name, version))

    def _add_replica(self, name, version, input_type, image, gpu_num=None, cpu_str=None, use_nvidia_docker=False):
        # containers = self.docker_client.containers.list(
        #     filters={"label": CLIPPER_QUERY_FRONTEND_CONTAINER_LABEL})
        # if len(containers) < 1:
        #     logger.warning("No Clipper query frontend found.")
        #     raise ClipperException(
        #         "No Clipper query frontend to attach model container to")
        query_frontend_hostname = self.query_frontend_name
        env_vars = {
            "CLIPPER_MODEL_NAME": name,
            "CLIPPER_MODEL_VERSION": version,
            # NOTE: assumes this container being launched on same machine
            # in same docker network as the query frontend
            "CLIPPER_IP": query_frontend_hostname,
            "CLIPPER_INPUT_TYPE": input_type,
        }
        labels = self.common_labels.copy()
        labels[CLIPPER_MODEL_CONTAINER_LABEL] = create_model_container_label(
            name, version)
        if use_nvidia_docker:
            # Even if a GPU-supported model isn't being deployed on a GPU,
            # we may still need to launch it using nvidia-docker because
            # the model framework may still depend on libcuda
            # os_env = os.environ.copy()
            cmd = ["nvidia-docker", "run", "-d",
                   "--network=%s" % self.docker_network]
            os_env = os.environ.copy()
            if gpu_num is not None:
                logger.info("Starting {name}:{version} on GPU {gpu_num}".format(
                    name=name, version=version, gpu_num=gpu_num))
                cmd.insert(0, "NV_GPU={}".format(str(gpu_num)))
                os_env["NV_GPU"] = str(gpu_num)

            else:
                # We're not running on a GPU, so we should mask all available
                # GPU resources
                cmd.append("-e")
                cmd.append("CUDA_VISIBLE_DEVICES=''")
            for k, v in labels.iteritems():
                cmd.append("-l")
                cmd.append("%s=%s" % (k, v))
            for k, v in env_vars.iteritems():
                cmd.append("-e")
                cmd.append("%s=%s" % (k, v))
            if cpu_str:
                cmd.append("--cpuset-cpus=%s" % cpu_str)
            # Mount logs dir
            cmd.append("-v")
            cmd.append("/home/ubuntu/logs:/logs")
            cmd.append(image)
            cmd_str = " ".join(cmd)
            logger.info("Docker command: \"%s\"" % cmd_str)
            self._execute(cmd_str)
            # subprocess.check_call(cmd, env=os_env)
        else:
            raise ClipperException("Model deployment requires nvidia docker")
            # self.docker_client.containers.run(
            #     image,
            #     environment=env_vars,
            #     labels=labels,
            #     cpuset_cpus=cpu_str,
            #     **self.extra_container_kwargs)

    def _add_replica_remote(self, name, version, input_type, image, remote_addr,
                            gpu_num=None, cpu_str=None, use_nvidia_docker=True):
        if self.host == "localhost":
            query_frontend_hostname = requests.get(
                "http://169.254.169.254/latest/meta-data/local-ipv4").text
        else:
            query_frontend_hostname = self.host
        env_vars = {
            "CLIPPER_MODEL_NAME": name,
            "CLIPPER_MODEL_VERSION": version,
            # NOTE: assumes this container being launched on same machine
            # in same docker network as the query frontend
            "CLIPPER_IP": query_frontend_hostname,
            "CLIPPER_INPUT_TYPE": input_type,
        }
        labels = self.common_labels.copy()
        labels[CLIPPER_MODEL_CONTAINER_LABEL] = create_model_container_label(
            name, version)
        if use_nvidia_docker:
            # Even if a GPU-supported model isn't being deployed on a GPU,
            # we may still need to launch it using nvidia-docker because
            # the model framework may still depend on libcuda
            # remote_env = {}
            cmd = ["nvidia-docker", "run", "-d"]
            if gpu_num is not None:
                logger.info("Starting {name}:{version} on GPU {gpu_num} on {remote_addr}".format(
                    name=name, version=version, gpu_num=gpu_num, remote_addr=remote_addr))
                cmd.insert(0, "NV_GPU={}".format(str(gpu_num)))
                # remote_env["NV_GPU"] = str(gpu_num)
            else:
                # We're not running on a GPU, so we should mask all available
                # GPU resources
                cmd.append("-e")
                cmd.append("CUDA_VISIBLE_DEVICES=''")
            for k, v in labels.iteritems():
                cmd.append("-l")
                cmd.append("%s=%s" % (k, v))
            for k, v in env_vars.iteritems():
                cmd.append("-e")
                cmd.append("%s=%s" % (k, v))
            if cpu_str:
                cmd.append("--cpuset-cpus=%s" % cpu_str)
            # Mount logs dir
            cmd.append("-v")
            cmd.append("/home/ubuntu/logs:/logs")
            cmd.append(image)
            logger.info("Docker command: \"%s\"" % cmd)
            env.host_string = remote_addr
            env.disable_known_hosts = True
            print(env.host_string)
            cmd_str = " ".join(cmd)
            # Don't use self._execute() here because if the query frontend host is local,
            # this will execute the command locally instead of remotely
            with show("everything"):
                run(cmd_str)
            # Don't forget to set hoststring back after launching the remote replica
            env.host_string = self.host
        else:
            raise ClipperException("Remote deployment requires nvidia docker")

    def set_num_replicas(self, name, version, input_type, image, num_replicas, **kwargs):
        """
        optional kwargs
        ----------
        allocated_cpus : list
            The set of PHYSICAL CPUs allocated to replicas of the deployed model
        cpus_per_replica : int
            The number of PHYSICAL CPUs allocated to each replica of the model
        """
        num_missing = num_replicas
        # logger.info(
        #     "Found {cur} replicas for {name}:{version}. Adding {missing}".
        #     format(
        #         cur=len(current_replicas),
        #         name=name,
        #         version=version,
        #         missing=(num_missing)))
        if "gpus" in kwargs:
            available_gpus = list(kwargs["gpus"])
        if "use_nvidia_docker" in kwargs:
            use_nvidia_docker = kwargs["use_nvidia_docker"]
        else:
            use_nvidia_docker = False

        # Enumerated list of cpus that can be allocated (e.g [1, 2, 3, 8, 9])
        if "allocated_cpus" in kwargs:
            allocated_cpus = kwargs["allocated_cpus"]
        if "cpus_per_replica" in kwargs:
            cpus_per_replica = kwargs["cpus_per_replica"]
        if (len(allocated_cpus) / cpus_per_replica) < num_missing:
            raise ClipperException(
                "Not enough cpus available. Trying to allocate {reps} replicas \
                {cpus_per} CPUs each out of only {alloc_cpus} allocated cpus".format(
                    reps=num_missing,
                    cpus_per=cpus_per_replica,
                    alloc_cpus=len(allocated_cpus)))
        for i in range(num_missing):
            if len(available_gpus) > 0:
                gpu_num = available_gpus.pop(0)
                use_nvidia_docker = True
            else:
                gpu_num = None
            cpus = allocated_cpus[i*cpus_per_replica: (i+1)*cpus_per_replica]
            cpus = self.get_virtual_cpus(cpus)
            cpus = [str(c) for c in cpus]
            cpu_str = ",".join(cpus)

            self._add_replica(name, version, input_type, image, gpu_num=gpu_num,
                              cpu_str=cpu_str, use_nvidia_docker=use_nvidia_docker)

        # elif len(current_replicas) > num_replicas:
        #     num_extra = len(current_replicas) - num_replicas
        #     logger.info(
        #         "Found {cur} replicas for {name}:{version}. Removing {extra}".
        #         format(
        #             cur=len(current_replicas),
        #             name=name,
        #             version=version,
        #             extra=(num_extra)))
        #     while len(current_replicas) > num_replicas:
        #         cur_container = current_replicas.pop()
        #         cur_container.stop()

    def set_num_replicas_remote(self, name, version, input_type, image, remote_addr, num_replicas,
                                **kwargs):
        """
        optional kwargs
        ----------
        allocated_cpus : list
            The set of PHYSICAL CPUs allocated to replicas of the deployed model
        cpus_per_replica : int
            The number of PHYSICAL CPUs allocated to each replica of the model
        """
        # current_replicas = self._get_replicas(name, version)
        # if len(current_replicas) < num_replicas:
        if True:
            num_missing = num_replicas
            if "gpus" in kwargs:
                available_gpus = list(kwargs["gpus"])
            use_nvidia_docker = True

            # Enumerated list of cpus that can be allocated (e.g [1, 2, 3, 8, 9])
            if "allocated_cpus" in kwargs:
                allocated_cpus = kwargs["allocated_cpus"]
            if "cpus_per_replica" in kwargs:
                cpus_per_replica = kwargs["cpus_per_replica"]
            if (len(allocated_cpus) / cpus_per_replica) < num_missing:
                raise ClipperException(
                    "Not enough cpus available. Trying to allocate {reps} replicas \
                    {cpus_per} CPUs each out of only {alloc_cpus} allocated cpus".format(
                        reps=num_missing,
                        cpus_per=cpus_per_replica,
                        alloc_cpus=len(allocated_cpus)))
            for i in range(num_missing):
                if len(available_gpus) > 0:
                    gpu_num = available_gpus.pop(0)
                    use_nvidia_docker = True
                else:
                    gpu_num = None
                cpus = allocated_cpus[i*cpus_per_replica: (i+1)*cpus_per_replica]
                cpus = self.get_virtual_cpus(cpus)
                cpus = [str(c) for c in cpus]
                cpu_str = ",".join(cpus)

                self._add_replica_remote(name, version, input_type, image, remote_addr=remote_addr, gpu_num=gpu_num,
                                  cpu_str=cpu_str, use_nvidia_docker=use_nvidia_docker)

    def get_virtual_cpus(self, pcpus):
        """
        Given a list of physical cpus,
        obtains a list of corresponding virtual cpus.

        NOTE: This method assumes a p2.8xlarge instance
        """
        vcpus_map = {}
        for i in range(16):
            vcpus_map[i] = (i, i + 16)

        vcpus = []
        for pcpu in pcpus:
            vcpus += list(vcpus_map[pcpu])

        return vcpus


    def get_logs(self, logging_dir):
        return None
        # containers = self.docker_client.containers.list(
        #     filters={"label": CLIPPER_DOCKER_LABEL})
        # logging_dir = os.path.abspath(os.path.expanduser(logging_dir))
        #
        # log_files = []
        # if not os.path.exists(logging_dir):
        #     os.makedirs(logging_dir)
        #     logger.info("Created logging directory: %s" % logging_dir)
        # for c in containers:
        #     log_file_name = "image_{image}:container_{id}.log".format(
        #         image=c.image.short_id, id=c.short_id)
        #     log_file = os.path.join(logging_dir, log_file_name)
        #     with open(log_file, "w") as lf:
        #         lf.write(c.logs(stdout=True, stderr=True))
        #     log_files.append(log_file)
        # return log_files

    def stop_models(self, models):
        pass
        # containers = self.docker_client.containers.list(
        #     filters={"label": CLIPPER_MODEL_CONTAINER_LABEL})
        # for c in containers:
        #     c_name, c_version = parse_model_container_label(
        #         c.labels[CLIPPER_MODEL_CONTAINER_LABEL])
        #     if c_name in models and c_version in models[c_name]:
        #         c.stop()

    def stop_all_model_containers(self):
        pass
        # containers = self.docker_client.containers.list(
        #     filters={"label": CLIPPER_MODEL_CONTAINER_LABEL})
        # for c in containers:
        #     c.stop()

    def stop_all(self, remote_addrs=None):
        """

        Parameters
        ----------
        remote_addrs : list(str)
            List of remote machine addresses to stop docker containers on
        """
        self._execute("docker stop $(docker ps -aq --filter label={})".format(CLIPPER_DOCKER_LABEL), warn_only=True)
        # containers = self.docker_client.containers.list(
        #     filters={"label": CLIPPER_DOCKER_LABEL})
        # for c in containers:
        #     c.stop(timeout=1)
        # try:
        #     self.docker_client.containers.prune()
        # except docker.errors.APIError as e:
        #     pass
        if remote_addrs is not None:
            logger.info("Stopping remote containers")
            for r in remote_addrs:
                env.host_string = r
                run("docker stop $(docker ps -aq --filter label={})".format(CLIPPER_DOCKER_LABEL))

        env.host_string = self.host

    def get_admin_addr(self):
        return "{host}:{port}".format(
            host=self.public_hostname, port=self.clipper_management_port)

    def get_query_addr(self):
        return "{host}:{port}".format(
            host=self.public_hostname, port=self.clipper_query_port)
