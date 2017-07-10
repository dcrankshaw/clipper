from __future__ import absolute_import, division, print_function
from ..container_manager import (ContainerManager, CLIPPER_DOCKER_LABEL,
                                 CLIPPER_MODEL_CONTAINER_LABEL)

from contextlib import contextmanager
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import logging
import json
import yaml


@contextmanager
def _pass_conflicts():
    try:
        yield
    except ApiException as e:
        body = json.loads(e.body)
        if body['reason'] == 'AlreadyExists':
            logging.info(
                "{} already exists, skipping!".format(body['details']))
            pass


class K8sContainerManager(ContainerManager):
    def __init__(self, clipper_public_hostname, create_registry=False):
        super.__init__(clipper_public_hostname)
        config.load_kube_config()
        self._k8s_v1 = client.CoreV1Api()
        self._k8s_beta = client.ExtensionsV1beta1Api()
        self.registry = None
        if create_registry:
            self.registry = self._start_registry()

    def _start_registry(self):
        """

        Returns
        -------
        str
            The address of the registry
        """
        logging.info("Initializing Docker registry on k8s cluster")
        with _pass_conflicts():
            self._k8s_v1.create_namespaced_replication_controller(
                body=yaml.load(
                    open(
                        'kube-registry-replication-controller.yaml'
                    )),
                namespace='kube-system')
        with _pass_conflicts():
            self._k8s_v1.create_namespaced_service(
                body=yaml.load(
                    open('kube-registry-service.yaml')),
                namespace='kube-system')
        with _pass_conflicts():
            self._k8s_beta.create_namespaced_daemon_set(
                body=yaml.load(
                    open('kube-registry-daemon-set.yaml')
                ),
                namespace='kube-system')
        return "localhost:5000"

    def start_clipper(self):
        """Deploys Clipper to the k8s cluster and exposes the frontends as services."""
        logging.info("Initializing Clipper services to k8s cluster")
        for name in ['mgmt-frontend', 'query-frontend', 'redis']:
            with _pass_conflicts():
                self._k8s_beta.create_namespaced_deployment(
                    body=yaml.load(
                        open('clipper/{}-deployment.yaml'.format(name))),
                    namespace='default')
            with _pass_conflicts():
                self._k8s_v1.create_namespaced_service(
                    body=yaml.load(
                        open('clipper/{}-service.yaml'.format(name))),
                    namespace='default')

    def deploy_model(self, name, version, input_type, repo):
        """Deploys a versioned model to a k8s cluster.

        Parameters
        ----------
        name : str
            The name to assign this model.
        version : int
            The version to assign this model.
        repo : str
            A docker repository path, which must be accessible by the k8s cluster.
        """
        with _pass_conflicts():
            # TODO: handle errors where `repo` is not accessible
            self._k8s_beta.create_namespaced_deployment(
                body={
                    'apiVersion': 'extensions/v1beta1',
                    'kind': 'Deployment',
                    'metadata': {
                        'name': name +
                        '-deployment'  # NOTE: must satisfy RFC 1123 pathname conventions
                    },
                    'spec': {
                        'replicas': 1,
                        'template': {
                            'metadata': {
                                'labels': {
                                    CLIPPER_MODEL_CONTAINER_LABEL:
                                    '',
                                    'model':
                                    name,
                                    'version':
                                    str(version)
                                }
                            },
                            'spec': {
                                'containers': [{
                                    'name':
                                    name,
                                    'image':
                                    repo,
                                    'ports': [{
                                        'containerPort': 80
                                    }],
                                    'env': [{
                                        'name': 'CLIPPER_MODEL_NAME',
                                        'value': name
                                    }, {
                                        'name': 'CLIPPER_MODEL_VERSION',
                                        'value': str(version)
                                    }, {
                                        'name': 'CLIPPER_IP',
                                        'value': 'query-frontend'
                                    }]
                                }]
                            }
                        }
                    }
                },
                namespace='default')

    def add_replica(self, name, version, input_type, repo):
        # TODO(feynman): Implement this
        pass

    def get_logs(self, logging_dir):
        # TODO(feynman): Implement this
        pass

    def stop_models(self, model_name=None, keep_version=None):
        # TODO(feynman): Account for model_name and keep_version.
        # NOTE: the format of the value of CLIPPER_MODEL_CONTAINER_LABEL
        # is "model_name:model_version"
        """Stops all deployments of pods running Clipper models."""
        logging.info("Stopping all running Clipper model deployments")
        try:
            self._k8s_beta.delete_collection_namespaced_deployment(
                namespace='default',
                label_selector=CLIPPER_MODEL_CONTAINER_LABEL)
        except ApiException as e:
            logging.warn("Exception deleting k8s deployments: {}".format(e))

    def stop_clipper(self):
        """Stops all Clipper resources.

        WARNING: Data stored on an in-cluster Redis deployment will be lost!
        This method does not delete any existing in-cluster Docker registry.
        """
        logging.info("Stopping all running Clipper resources")

        try:
            for service in self._k8s_v1.list_namespaced_service(
                    namespace='default',
                    label_selector=CLIPPER_DOCKER_LABEL).items:
                # TODO: use delete collection of services if API provides
                service_name = service.metadata.name
                self._k8s_v1.delete_namespaced_service(
                    namespace='default', name=service_name)

            self._k8s_beta.delete_collection_namespaced_deployment(
                namespace='default',
                label_selector=CLIPPER_DOCKER_LABEL)

            self._k8s_v1.delete_collection_namespaced_pod(
                namespace='default',
                label_selector=CLIPPER_DOCKER_LABEL)

            self._k8s_v1.delete_collection_namespaced_pod(
                namespace='default',
                label_selector=CLIPPER_MODEL_CONTAINER_LABEL)
        except ApiException as e:
            logging.warn("Exception deleting k8s resources: {}".format(e))

    def get_registry(self):
        return self.registry
