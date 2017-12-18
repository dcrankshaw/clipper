from __future__ import absolute_import, division, print_function
# import docker
import googleapiclient.discovery
import logging
import os
import random
import time
from ..container_manager import (
    create_model_container_label, parse_model_container_label,
    ContainerManager, CLIPPER_DOCKER_LABEL, CLIPPER_MODEL_CONTAINER_LABEL,
    CLIPPER_QUERY_FRONTEND_CONTAINER_LABEL,
    CLIPPER_MGMT_FRONTEND_CONTAINER_LABEL, CLIPPER_INTERNAL_RPC_PORT,
    CLIPPER_INTERNAL_MANAGEMENT_PORT)
from ..exceptions import ClipperException
import subprocess32 as subprocess

logger = logging.getLogger(__name__)

PROJECT_ID = "clipper-model-comp"


class GCPContainerManager(ContainerManager):
    def __init__(self, cluster_name):
        self.project = "clipper-model-comp"
        self.zone = "us-east1-b"
        self.cluster_name = cluster_name
        self.compute = googleapiclient.discovery.build('compute', 'v1')

    def start_redis(self):
        redis_config = {
                "name": "redis-{}".format(self.cluster_name),
                "zone": "projects/clipper-model-comp/zones/us-east1-b",
                "minCpuPlatform": "Automatic",
                "machineType": "projects/clipper-model-comp/zones/us-east1-b/machineTypes/n1-standard-1",
                "metadata": {
                    "items": [
                        {
                            "key": "gce-container-declaration",
                            "value": "spec:\n  containers:\n    - name: redis-{cluster_name}\n      image: 'gcr.io/clipper-model-comp/redis:alpine'\n      stdin: false\n      tty: false\n  restartPolicy: Always\n".format(cluster_name=self.cluster_name)
                            }
                        ]
                    },
                "tags": {
                    "items": []
                    },
                "disks": [
                    {
                        "type": "PERSISTENT",
                        "boot": True,
                        "mode": "READ_WRITE",
                        "autoDelete": True,
                        "deviceName": "redis-{}".format(self.cluster_name),
                        "initializeParams": {
                            "sourceImage": "projects/cos-cloud/global/images/cos-stable-63-10032-71-0",
                            "diskType": "projects/clipper-model-comp/zones/us-east1-b/diskTypes/pd-standard",
                            "diskSizeGb": "10"
                            }
                        }
                    ],
                "canIpForward": False,
                "networkInterfaces": [
                    {
                        "network": "projects/clipper-model-comp/global/networks/default",
                        "subnetwork": "projects/clipper-model-comp/regions/us-east1/subnetworks/default",
                        "accessConfigs": [
                            {
                                "name": "External NAT",
                                "type": "ONE_TO_ONE_NAT"
                                }
                            ],
                        "aliasIpRanges": []
                        }
                    ],
                "description": "",
                "labels": {
                    "container-vm": "cos-stable-63-10032-71-0",
                    "clipper-cluster": self.cluster_name 
                    },
                "scheduling": {
                    "preemptible": False,
                    "onHostMaintenance": "MIGRATE",
                    "automaticRestart": True
                    },
                "deletionProtection": False,
            "serviceAccounts": [
                    {
                        "email": "450655029092-compute@developer.gserviceaccount.com",
                        "scopes": [
                            "https://www.googleapis.com/auth/devstorage.read_only",
                            "https://www.googleapis.com/auth/logging.write",
                            "https://www.googleapis.com/auth/monitoring.write",
                            "https://www.googleapis.com/auth/servicecontrol",
                            "https://www.googleapis.com/auth/service.management.readonly",
                            "https://www.googleapis.com/auth/trace.append"
                            ]
                        }
                    ]
            }

        self.start_instance(redis_config)
        instances = self.compute.instances().list(project=self.project, zone=self.zone).execute()
        self.redis_ip = None
        for inst in instances["items"]:
            if inst["name"] == "redis-{}".format(self.cluster_name):
                self.redis_ip = inst["networkInterfaces"][0]["networkIP"]
                logger.info("Setting redis IP to {}".format(self.redis_ip))
                break
        if self.redis_ip is None:
            logger.error("No Redis instance found")

        
    def start_mgmt_frontend(self):
        mgmt_config = {
          "name": "clipper-mgmt-{}".format(self.cluster_name),
          "zone": "projects/clipper-model-comp/zones/us-east1-b",
          "minCpuPlatform": "Automatic",
          "machineType": "projects/clipper-model-comp/zones/us-east1-b/machineTypes/n1-standard-1",
          "metadata": {
            "items": [
              {
                "key": "gce-container-declaration",
                "value": "spec:\n  containers:\n    - name: clipper-mgmt-{cluster_name}\n      image: 'gcr.io/clipper-model-comp/management_frontend:develop'\n      args:\n        - '--redis_ip={redis_ip}'\n        - '--redis_port=6379'\n      stdin: false\n      tty: false\n  restartPolicy: Always\n".format(cluster_name=self.cluster_name, redis_ip=self.redis_ip)
              }
            ]
          },
          "tags": {
            "items": []
          },
          "disks": [
            {
              "type": "PERSISTENT",
              "boot": True,
              "mode": "READ_WRITE",
              "autoDelete": True,
              "deviceName": "clipper-mgmt-{}".format(self.cluster_name),
              "initializeParams": {
                "sourceImage": "https://www.googleapis.com/compute/v1/projects/cos-cloud/global/images/cos-stable-63-10032-71-0",
                "diskType": "projects/clipper-model-comp/zones/us-east1-b/diskTypes/pd-standard",
                "diskSizeGb": "10"
              }
            }
          ],
          "canIpForward": False,
          "networkInterfaces": [
            {
              "network": "projects/clipper-model-comp/global/networks/default",
              "subnetwork": "projects/clipper-model-comp/regions/us-east1/subnetworks/default",
              "accessConfigs": [
                {
                  "name": "External NAT",
                  "type": "ONE_TO_ONE_NAT"
                }
              ],
              "aliasIpRanges": []
            }
          ],
          "description": "",
          "labels": {
            "container-vm": "cos-stable-63-10032-71-0",
            "clipper-cluster": self.cluster_name 
          },
          "scheduling": {
            "preemptible": False,
            "onHostMaintenance": "MIGRATE",
            "automaticRestart": True
          },
          "deletionProtection": False,
          "serviceAccounts": [
            {
              "email": "450655029092-compute@developer.gserviceaccount.com",
              "scopes": [
                "https://www.googleapis.com/auth/devstorage.read_only",
                "https://www.googleapis.com/auth/logging.write",
                "https://www.googleapis.com/auth/monitoring.write",
                "https://www.googleapis.com/auth/servicecontrol",
                "https://www.googleapis.com/auth/service.management.readonly",
                "https://www.googleapis.com/auth/trace.append"
              ]
            }
          ]
        }

        self.start_instance(mgmt_config)

    def start_query_frontend(self):
        query_config = {
          "name": "clipper-query-{}".format(self.cluster_name),
          "zone": "projects/clipper-model-comp/zones/us-east1-b",
          "minCpuPlatform": "Automatic",
          "machineType": "projects/clipper-model-comp/zones/us-east1-b/machineTypes/custom-4-56320-ext",
          "metadata": {
            "items": [
              {
                "key": "gce-container-declaration",
                "value": "spec:\n  containers:\n    - name: clipper-query-{cluster_name}\n      image: 'gcr.io/clipper-model-comp/zmq_frontend:develop'\n      args:\n        - '--redis_ip={redis_ip}'\n        - '--redis_port=6379'\n      stdin: false\n      tty: false\n  restartPolicy: Always\n".format(cluster_name=self.cluster_name, redis_ip=self.redis_ip)
              }
            ]
          },
          "tags": {
            "items": []
          },
          "disks": [
            {
              "type": "PERSISTENT",
              "boot": True,
              "mode": "READ_WRITE",
              "autoDelete": True,
              "deviceName": "clipper-query-{}".format(self.cluster_name),
              "initializeParams": {
                "sourceImage": "https://www.googleapis.com/compute/v1/projects/cos-cloud/global/images/cos-stable-63-10032-71-0",
                "diskType": "projects/clipper-model-comp/zones/us-east1-b/diskTypes/pd-standard",
                "diskSizeGb": "10"
              }
            }
          ],
          "canIpForward": False,
          "networkInterfaces": [
            {
              "network": "projects/clipper-model-comp/global/networks/default",
              "subnetwork": "projects/clipper-model-comp/regions/us-east1/subnetworks/default",
              "accessConfigs": [
                {
                  "name": "External NAT",
                  "type": "ONE_TO_ONE_NAT"
                }
              ],
              "aliasIpRanges": []
            }
          ],
          "description": "",
          "labels": {
            "container-vm": "cos-stable-63-10032-71-0",
            "clipper-cluster": self.cluster_name 
          },
          "scheduling": {
            "preemptible": False,
            "onHostMaintenance": "MIGRATE",
            "automaticRestart": True
          },
          "deletionProtection": False,
          "serviceAccounts": [
            {
              "email": "450655029092-compute@developer.gserviceaccount.com",
              "scopes": [
                "https://www.googleapis.com/auth/devstorage.read_only",
                "https://www.googleapis.com/auth/logging.write",
                "https://www.googleapis.com/auth/monitoring.write",
                "https://www.googleapis.com/auth/servicecontrol",
                "https://www.googleapis.com/auth/service.management.readonly",
                "https://www.googleapis.com/auth/trace.append"
              ]
            }
          ]
        }

        self.start_instance(query_config)

    def start_instance(self, config):
        op = self.compute.instances().insert(project=self.project, zone=self.zone, body=config).execute()
        while True:
            result = self.compute.zoneOperations().get(
                    project=self.project, zone=self.zone, operation=op['name']).execute()
            if result['status'] == 'DONE':
                break
            else:
                time.sleep(1)

    def start_clipper(self, *args):
        self.start_redis()
        self.start_mgmt_frontend()
        self.start_query_frontend()
        self.connect()

    def connect(self):
        instances = self.compute.instances().list(project=self.project, zone=self.zone).execute()
        for inst in instances["items"]:
            if inst["name"] == "clipper-query-{}".format(self.cluster_name):
                self.query_frontend_ip = inst["networkInterfaces"][0]["networkIP"]
                logger.info("Setting ZMQ frontend IP to {}".format(self.query_frontend_ip))
                break
        if self.query_frontend_ip is None:
            logger.error("No query frontend instance found")


    def deploy_model(self, name, version, input_type, image, num_replicas=1, **kwargs):
        # Parameters
        # ----------
        # image : str
        #     The fully specified Docker imagesitory to deploy. If using a custom
        #     registry, the registry name must be prepended to the image. For example,
        #     "localhost:5000/my_model_name:my_model_version" or
        #     "quay.io/my_namespace/my_model_name:my_model_version"
        self.set_num_replicas(name, version, input_type, image, num_replicas, **kwargs)

    def _get_replicas(self, name, version):
        containers = self.docker_client.containers.list(
            filters={
                "label":
                "{key}={val}".format(
                    key=CLIPPER_MODEL_CONTAINER_LABEL,
                    val=create_model_container_label(name, version))
            })
        return containers

    def get_num_replicas(self, name, version):
        return len(self._get_replicas(name, version))

    def _add_replica(self, name, version, input_type, image, gpu_num=None, cpu_str=None, use_nvidia_docker=False):
        containers = self.docker_client.containers.list(
            filters={"label": CLIPPER_QUERY_FRONTEND_CONTAINER_LABEL})
        if len(containers) < 1:
            logger.warning("No Clipper query frontend found.")
            raise ClipperException(
                "No Clipper query frontend to attach model container to")
        query_frontend_hostname = containers[0].name
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
            env = os.environ.copy()
            cmd = ["nvidia-docker", "run", "-d",
                   "--network=%s" % self.docker_network]
            if gpu_num is not None:
                logger.info("Starting {name}:{version} on GPU {gpu_num}".format(
                    name=name, version=version, gpu_num=gpu_num))
                env["NV_GPU"] = str(gpu_num)
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
            cmd.append(image)
            logger.info("Docker command: \"%s\"" % cmd)
            subprocess.check_call(cmd, env=env)
        else:
            self.docker_client.containers.run(
                image,
                environment=env_vars,
                labels=labels,
                cpuset_cpus=cpu_str,
                **self.extra_container_kwargs)

    def set_num_replicas(self, name, version, input_type, image, num_replicas, **kwargs):
        current_replicas = self._get_replicas(name, version)
        if len(current_replicas) < num_replicas:
            num_missing = num_replicas - len(current_replicas)
            logger.info(
                "Found {cur} replicas for {name}:{version}. Adding {missing}".
                format(
                    cur=len(current_replicas),
                    name=name,
                    version=version,
                    missing=(num_missing)))
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
                    gpu_num = available_gpus.pop()
                    use_nvidia_docker = True
                else:
                    gpu_num = None
                cpus = allocated_cpus[i*cpus_per_replica: (i+1)*cpus_per_replica]
                cpus = [str(c) for c in cpus]
                cpu_str = ",".join(cpus)

                self._add_replica(name, version, input_type, image, gpu_num=gpu_num,
                                  cpu_str=cpu_str, use_nvidia_docker=use_nvidia_docker)
        elif len(current_replicas) > num_replicas:
            num_extra = len(current_replicas) - num_replicas
            logger.info(
                "Found {cur} replicas for {name}:{version}. Removing {extra}".
                format(
                    cur=len(current_replicas),
                    name=name,
                    version=version,
                    extra=(num_extra)))
            while len(current_replicas) > num_replicas:
                cur_container = current_replicas.pop()
                cur_container.stop()

    def get_logs(self, logging_dir):
        containers = self.docker_client.containers.list(
            filters={"label": CLIPPER_DOCKER_LABEL})
        logging_dir = os.path.abspath(os.path.expanduser(logging_dir))

        log_files = []
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
            logger.info("Created logging directory: %s" % logging_dir)
        for c in containers:
            log_file_name = "image_{image}:container_{id}.log".format(
                image=c.image.short_id, id=c.short_id)
            log_file = os.path.join(logging_dir, log_file_name)
            with open(log_file, "w") as lf:
                lf.write(c.logs(stdout=True, stderr=True))
            log_files.append(log_file)
        return log_files

    def stop_models(self, models):
        containers = self.docker_client.containers.list(
            filters={"label": CLIPPER_MODEL_CONTAINER_LABEL})
        for c in containers:
            c_name, c_version = parse_model_container_label(
                c.labels[CLIPPER_MODEL_CONTAINER_LABEL])
            if c_name in models and c_version in models[c_name]:
                c.stop()

    def stop_all_model_containers(self):
        containers = self.docker_client.containers.list(
            filters={"label": CLIPPER_MODEL_CONTAINER_LABEL})
        for c in containers:
            c.stop()

    def stop_all(self):
        containers = self.docker_client.containers.list(
            filters={"label": CLIPPER_DOCKER_LABEL})
        for c in containers:
            c.stop(timeout=1)
        try:
            self.docker_client.containers.prune()
        except docker.errors.APIError as e:
            pass

    def get_admin_addr(self):
        return "{host}:{port}".format(
            host=self.public_hostname, port=self.clipper_management_port)

    def get_query_addr(self):
        return "{host}:{port}".format(
            host=self.public_hostname, port=self.clipper_query_port)
