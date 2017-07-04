import docker
import logging

from container_manager import *



DOCKER_NETWORK_NAME = "clipper_network"

class DockerContainerManager(ContainerManager):

    def __init__(self, clipper_public_hostname, redis_ip=None, redis_port=6379, extra_container_kwargs={}):
        """
        Parameters
        ----------
        clipper_public_hostname : str
            The public hostname or IP address at which the Clipper Docker
            containers can be accessed via their exposed ports. On macOs this
            can be set to "localhost" as Docker automatically makes exposed
            ports on Docker containers available via localhost, but on other
            operating systems this must be set explicitly.

        extra_container_kwargs : dict
            Any additional keyword arguments to pass to the call to
            :py:meth:`docker.client.containers.run`.
        """
        self.docker_client = docker.from_env()
        self.redis_port = redis_port
        self.redis_ip = redis_ip
        self.extra_container_kwargs = extra_container_kwargs
        self.public_hostname = clipper_public_hostname

        #TODO: Deal with Redis persistence
    
    def start_clipper(self):
        self.host_ip = host_ip
        self.docker_client.networks.create(DOCKER_NETWORK_NAME)
        container_args = {
                "network": DOCKER_NETWORK_NAME,
                "labels" : [CLIPPER_DOCKER_LABEL],
                "detach": True,
                }
        self.extra_container_kwargs.update(container_args)


        if self.redis_ip = None:
            logging.info("Starting managed Redis instance in Docker")
            self.redis_ip = "redis"
            self.docker_client.containers.run(
                    'redis:alpine',
                    "redis-server --port %d" % self.redis_port,
                    name="redis",
                    ports={'%s/tcp' % self.redis_port: self.redis_port}
                    **self.extra_container_kwargs)



        cmd = "--redis_ip={redis_ip} --redis_port={redis_port}".format(redis_ip=self.redis_ip, redis_port=self.redis_port)
        self.docker_client.containers.run(
                'clipper/management_frontend:latest',
                cmd,
                name="mgmt_frontend",
                ports={'%s/tcp' % CLIPPER_MANAGEMENT_PORT: CLIPPER_MANAGEMENT_PORT}
                **self.extra_container_kwargs)
        self.docker_client.containers.run(
                'clipper/query_frontend:latest',
                cmd,
                name="query_frontend",
                ports={'%s/tcp' % CLIPPER_QUERY_PORT: CLIPPER_QUERY_PORT}
                **self.extra_container_kwargs)

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

    def get_admin_addr():
        return "{host}:{port}".format(host=self.public_hostname, port=CLIPPER_MANAGEMENT_PORT)
