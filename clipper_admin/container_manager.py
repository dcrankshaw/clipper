import abc

# Constants

CLIPPER_QUERY_PORT = 1337
CLIPPER_MANAGEMENT_PORT = 1338
CLIPPER_RPC_PORT = 7000

CLIPPER_DOCKER_LABEL = "ai.clipper.container.label"
CLIPPER_MODEL_CONTAINER_LABEL = "ai.clipper.model_container.label"
CONTAINERLESS_MODEL_IMAGE = "NO_CONTAINER"

class ContainerManager(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, clipper_public_hostname):
        self.public_hostname = clipper_public_hostname

    
    @abc.abstractmethod
    def start_clipper(self):
        return

    # TODO(feynmanliang): Do we need a separate deploy_model method?
    # It seems like in k8s, you might create a deploymentservice
    # when a model is deployed, and then just add replicas to it
    # when  calling `add_replica()`. In this case, the two methods
    # would require different behavior.
    @abc.abstractmethod
    def deploy_model(self, name, version, input_type, repo):
        return

    @abc.abstractmethod
    def add_replica(self, name, version, input_type, repo):
        return

    @abc.abstractmethod
    def get_logs(self, logging_dir):
        return

    @abc.abstractmethod
    def stop_models(self):
        return

    @abc.abstractmethod
    def stop_clipper(self):
        pass

    def get_admin_addr(self):
        return "{host}:{port}".format(host=self.public_hostname, port=CLIPPER_MANAGEMENT_PORT)

    def get_registry(self):
        """Return a reference to the Docker registry created by the ContainerManager
        """
        return None

