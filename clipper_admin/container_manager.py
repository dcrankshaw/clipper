import abc

# Constants

CLIPPER_QUERY_PORT = 1337
CLIPPER_MANAGEMENT_PORT = 1338
CLIPPER_RPC_PORT = 7000

CLIPPER_DOCKER_LABEL = "ai.clipper.container.label"
CLIPPER_MODEL_CONTAINER_LABEL = "ai.clipper.model_container.label"

class ContainerManager(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def start_clipper():
        pass

    @abc.abstractmethod
    def deploy_model():
        pass

    @abc.abstractmethod
    def add_container():
        pass

    @abc.abstractmethod
    def get_container_logs(self):
        pass

    @abc.abstractmethod
    def stop_models():
        pass

    @abc.abstractmethod
    def stop_clipper():
        pass

    @abc.abstractmethod
    def get_admin_addr():
        pass

