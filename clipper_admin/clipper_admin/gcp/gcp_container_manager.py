from __future__ import absolute_import, division, print_function
# import docker
import googleapiclient.discovery
import logging
import os
import random
import time
import numpy as np
from ..container_manager import (
    create_model_container_label, parse_model_container_label,
    ContainerManager, CLIPPER_DOCKER_LABEL, CLIPPER_MODEL_CONTAINER_LABEL,
    CLIPPER_QUERY_FRONTEND_CONTAINER_LABEL,
    CLIPPER_INTERNAL_QUERY_PORT,
    CLIPPER_MGMT_FRONTEND_CONTAINER_LABEL, CLIPPER_INTERNAL_RPC_PORT,
    CLIPPER_INTERNAL_MANAGEMENT_PORT)
from ..exceptions import ClipperException
import subprocess32 as subprocess

logger = logging.getLogger(__name__)
logging.getLogger('googleapiclient').setLevel(logging.ERROR)

PROJECT_ID = "clipper-model-comp"

class GCPContainerManager(ContainerManager):
    def __init__(self, cluster_name):
        self.project = "clipper-model-comp"
        self.zone = "us-west1-b"
        self.cluster_name = cluster_name
        self.compute = googleapiclient.discovery.build('compute', 'v1')
        self.redis_port = 6380

    def start_redis(self):
        startup_script = ("#! /bin/bash\ngcloud docker --authorize-only\ndocker run -d "
                          "--log-driver=gcplogs --log-opt gcp-log-cmd=true "
                          "-p {redis_port}:6379 {image}").format(image="redis:alpine", redis_port=self.redis_port)

        redis_config = {
                "name": "redis-{}".format(self.cluster_name),
                "zone": "projects/clipper-model-comp/zones/us-west1-b",
                "minCpuPlatform": "Automatic",
                "machineType": "projects/clipper-model-comp/zones/us-west1-b/machineTypes/n1-standard-1",
                "metadata": {
                    "items": [
                            {
                            "key": "startup-script",
                            "value": startup_script
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
                                "sourceImage": "projects/clipper-model-comp/global/images/clipper-core-docker",
                                "diskType": "projects/clipper-model-comp/zones/us-west1-b/diskTypes/pd-standard",
                                "diskSizeGb": "50"
                            }
                        }
                    ],
                "canIpForward": False,
                "networkInterfaces": [
                    {
                        "network": "projects/clipper-model-comp/global/networks/default",
                        "subnetwork": "projects/clipper-model-comp/regions/us-west1/subnetworks/default",
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

        self._start_instance(redis_config)
        instances = self.compute.instances().list(project=self.project, zone=self.zone).execute()
        self.redis_internal_ip = None
        for inst in instances["items"]:
            if inst["name"] == "redis-{}".format(self.cluster_name):
                self.redis_internal_ip = inst["networkInterfaces"][0]["networkIP"]
                self.redis_external_ip = inst["networkInterfaces"][0]["accessConfigs"][0]["natIP"]
                logger.info("Setting redis IP to {}".format(self.redis_internal_ip))
                break
        if self.redis_internal_ip is None:
            logger.error("No Redis instance found")


    def start_mgmt_frontend(self):
        startup_script = ("#! /bin/bash\ngcloud docker --authorize-only\ndocker run -d "
                          "--log-driver=gcplogs --log-opt gcp-log-cmd=true "
                          "-p {mgmt_port}:{mgmt_port} {image} --redis_ip={redis_ip} --redis_port={redis_port}"
                          ).format(mgmt_port=CLIPPER_INTERNAL_MANAGEMENT_PORT,
                                   image="gcr.io/clipper-model-comp/management_frontend:debug",
                                   redis_ip=self.redis_internal_ip, redis_port=self.redis_port)

        mgmt_config = {
          "name": "clipper-mgmt-{}".format(self.cluster_name),
          "zone": "projects/clipper-model-comp/zones/us-west1-b",
          "minCpuPlatform": "Automatic",
          "machineType": "projects/clipper-model-comp/zones/us-west1-b/machineTypes/n1-standard-1",
          "metadata": {
            "items": [
                {
                "key": "startup-script",
                "value": startup_script
                }
            ]
          },
          "tags": {
            "items": [
                    "clipper-mgmt"
                ]
          },
          "disks": [
            {
              "type": "PERSISTENT",
              "boot": True,
              "mode": "READ_WRITE",
              "autoDelete": True,
              "deviceName": "clipper-mgmt-{}".format(self.cluster_name),
              "initializeParams": {
                    "sourceImage": "projects/clipper-model-comp/global/images/clipper-core-docker",
                    "diskType": "projects/clipper-model-comp/zones/us-west1-b/diskTypes/pd-standard",
                    "diskSizeGb": "50"
              }
            }
          ],
          "canIpForward": False,
          "networkInterfaces": [
            {
              "network": "projects/clipper-model-comp/global/networks/default",
              "subnetwork": "projects/clipper-model-comp/regions/us-west1/subnetworks/default",
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

        self._start_instance(mgmt_config)

    def start_query_frontend(self):

        startup_script = ("#! /bin/bash\ngcloud docker --authorize-only\ndocker run -d "
                          "--log-driver=gcplogs --log-opt gcp-log-cmd=true "
                          "-p 4455:4455 -p 4456:4456 -p 1337:1337 -p 7000:7000 "
                          "-p 7010:7010 -p 7011:7011 "
                          "{image} --redis_ip={redis_ip} --redis_port={redis_port}").format(
                                  image="gcr.io/clipper-model-comp/zmq_frontend:develop",
                                  redis_ip=self.redis_internal_ip, redis_port=self.redis_port)




        query_config = {
          "name": "clipper-query-{}".format(self.cluster_name),
          "zone": "projects/clipper-model-comp/zones/us-west1-b",
          "minCpuPlatform": "Automatic",
          "machineType": "projects/clipper-model-comp/zones/us-west1-b/machineTypes/custom-4-56320-ext",
          "metadata": {
            "items": [
                {
                "key": "startup-script",
                "value": startup_script
                }
            ]
          },
          "tags": {
            "items": [
                    "clipper-zmq-frontend"
                ]
          },
          "disks": [
            {
              "type": "PERSISTENT",
              "boot": True,
              "mode": "READ_WRITE",
              "autoDelete": True,
              "deviceName": "clipper-query-{}".format(self.cluster_name),
              "initializeParams": {
                # "sourceImage": "https://www.googleapis.com/compute/v1/projects/cos-cloud/global/images/cos-stable-63-10032-71-0",
                "sourceImage": "projects/clipper-model-comp/global/images/clipper-core-docker",
                "diskType": "projects/clipper-model-comp/zones/us-west1-b/diskTypes/pd-standard",
                "diskSizeGb": "100"
              }
            }
          ],
          "canIpForward": False,
          "networkInterfaces": [
            {
              "network": "projects/clipper-model-comp/global/networks/default",
              "subnetwork": "projects/clipper-model-comp/regions/us-west1/subnetworks/default",
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

        self._start_instance(query_config)

    def _wait_for_op(self, op):
        while True:
            result = self.compute.zoneOperations().get(
                    project=self.project, zone=self.zone, operation=op['name']).execute()
            if result['status'] == 'DONE':
                break
            else:
                time.sleep(1)

    def _start_instance(self, config):
        op = self.compute.instances().insert(project=self.project, zone=self.zone, body=config).execute()
        self._wait_for_op(op)

    def _delete_instance(self, rep_name):
        op = self.compute.instances().delete(project=self.project, zone=self.zone, instance=rep_name).execute()
        return op

    def start_clipper(self, *args):
        self.start_redis()
        self.start_mgmt_frontend()
        self.start_query_frontend()
        self.connect()

    def connect(self):
        instances = self.compute.instances().list(project=self.project, zone=self.zone).execute()
        for inst in instances["items"]:
            if inst["name"] == "clipper-query-{}".format(self.cluster_name):
                self.query_frontend_internal_ip = inst["networkInterfaces"][0]["networkIP"]
                self.query_frontend_external_ip = inst["networkInterfaces"][0]["accessConfigs"][0]["natIP"]
                logger.info("Setting ZMQ frontend internal IP to {}".format(self.query_frontend_internal_ip))
            if inst["name"] == "clipper-mgmt-{}".format(self.cluster_name):
                self.mgmt_frontend_internal_ip = inst["networkInterfaces"][0]["networkIP"]
                self.mgmt_frontend_external_ip = inst["networkInterfaces"][0]["accessConfigs"][0]["natIP"]
        if self.query_frontend_internal_ip is None:
            logger.error("No query frontend instance found")
        if self.mgmt_frontend_internal_ip is None:
            logger.error("No mgmt frontend instance found")


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
        replicas = self.compute.instances().list(project=self.project, zone=self.zone,
                filter="labels.clipper-model eq {name}-{version}".format(name=name, version=version)).execute()
        if "items" in replicas:
            return replicas["items"]
        else:
            return []

    def get_num_replicas(self, name, version):
        return len(self._get_replicas(name, version))

    def _add_replica(self, name, version, input_type, image, gpu_type=None, num_cpus=1, **kwargs):
        """
        Parameters
        ----------
        gpu_type : str
            Either "p100" or "k80". Default is None.
        """


        rep_name = "{name}-{version}-{random}".format(name=name, version=version, random=np.random.randint(0, 100000))

        startup_script = ("#! /bin/bash\ngcloud docker --authorize-only\nnvidia-docker run -d "
                          "--log-driver=gcplogs --log-opt gcp-log-cmd=true "
                          "--log-opt env=CLIPPER_MODEL_NAME "
                          "--log-opt env=CLIPPER_MODEL_VERSION "
                          "--log-opt labels=rep_name "
                          "-e CLIPPER_MODEL_NAME={name} "
                          "-e CLIPPER_MODEL_VERSION={version} "
                          "-e CLIPPER_IP={ip} "
                          "-e CLIPPER_INPUT_TYPE={input_type} "
                          "-l rep_name={rep_name} "
                          "{image}").format(
                                  name=name,
                                  version=version,
                                  ip=self.query_frontend_internal_ip,
                                  input_type=input_type,
                                  rep_name=rep_name,
                                  image=image)

        logger.info("STARTUP SCRIPT:\n\n {script}".format(script=startup_script))
        mem = min(5120*4, 5120*num_cpus)

        config = {
                "name": rep_name,
                "zone": "projects/clipper-model-comp/zones/us-west1-b",
                "minCpuPlatform": "Automatic",
                "machineType": "projects/clipper-model-comp/zones/us-west1-b/machineTypes/custom-{num_cpus}-{mem}".format(num_cpus=num_cpus, mem=mem),
                "metadata": {
                    "items": [
                        {
                            "key": "startup-script",
                            "value": startup_script
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
                        "deviceName": rep_name,
                        "initializeParams": {
                                "sourceImage": "projects/clipper-model-comp/global/images/clipper-k80-docker",
                                "diskType": "projects/clipper-model-comp/zones/us-west1-b/diskTypes/pd-standard",
                                "diskSizeGb": "50"
                            }
                        }
                    ],
                "canIpForward": False,
                "networkInterfaces": [
                    {
                        "network": "projects/clipper-model-comp/global/networks/default",
                        "subnetwork": "projects/clipper-model-comp/regions/us-west1/subnetworks/default",
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
                    "clipper-cluster": self.cluster_name,
                    "clipper-model": "{name}-{version}".format(name=name, version=version)
                    },
                "scheduling": {
                        "preemptible": False,
                        "onHostMaintenance": "TERMINATE",
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

        if gpu_type is not None:
            if gpu_type in ["p100", "k80"]:
                config["guestAccelerators"] = [
                        {
                            "acceleratorType": "projects/clipper-model-comp/zones/us-west1-b/acceleratorTypes/nvidia-tesla-{gpu_type}".format(gpu_type=gpu_type),
                            "acceleratorCount": 1
                            }
                        ]
                config["disks"][0]["initializeParams"]["sourceImage"] = "projects/clipper-model-comp/global/images/clipper-{gpu_type}-docker".format(gpu_type=gpu_type)
            else:
                logger.error("{} is invalid gpu type. Starting replica without a GPU.".format(gpu_type))
        self._start_instance(config)

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
            for _ in range(num_missing):
                self._add_replica(name, version, input_type, image, **kwargs)
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
                cur_replica = current_replicas.pop()
                self._delete_instance(cur_replica["name"])

    def get_logs(self, logging_dir):
        raise NotImplementedError

    def stop_models(self, models):
        raise NotImplementedError

    def stop_all_model_containers(self):
        raise NotImplementedError

    def stop_all(self):
        replicas = self.compute.instances().list(project=self.project, zone=self.zone,
                filter="labels.clipper-cluster eq {}".format(self.cluster_name)).execute()
        ops = []
        if "items" in replicas:
            for rep in replicas["items"]:
                ops.append(self._delete_instance(rep["name"]))
        for op in ops:
            self._wait_for_op(op)

    def get_admin_addr(self):
        return "{host}:{port}".format(
            host=self.mgmt_frontend_external_ip, port=CLIPPER_INTERNAL_MANAGEMENT_PORT)

    def get_query_addr(self):
        # raise NotImplementedError
        return "{host}:{port}".format(
            host=self.query_frontend_external_ip, port=CLIPPER_INTERNAL_QUERY_PORT)
