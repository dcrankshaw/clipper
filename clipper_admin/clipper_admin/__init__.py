from __future__ import absolute_import

from .docker.docker_container_manager import DockerContainerManager
from .aws.aws_container_manager import AWSContainerManager
# from .gcp.gcp_container_manager import GCPContainerManager
from .kubernetes.kubernetes_container_manager import KubernetesContainerManager
from .clipper_admin import *
from . import deployers
from .version import __version__
from .exceptions import ClipperException, UnconnectedException
