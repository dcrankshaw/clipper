from __future__ import absolute_import, division, print_function
import logging
import abc

"""
subclasses of ContainerManager
- Docker local (use Docker Python SDK)
- Docker + SSH (raw string docker commands?)
  - By default, deploy all containers to same machine, but allow a host arg
- Kubernetes
"""

logging.basicConfig(level=logging.INFO)

class ClipperException(Exception):
    pass

def start_clipper(cm):
    try:
        cm.start_clipper()
        logging.info("Clipper is running")
        return True
    except ClipperException as e:
        logging.info(e.msg)
        return False
        

def register_application(cm, name, model, input_type, default_output, slo_micros):
    url = "http://{host}/admin/add_app".format(host=cm.get_admin_addr())
    req_json = json.dumps({
        "name": name,
        "candidate_model_names": [model],
        "input_type": input_type,
        "default_output": default_output,
        "latency_slo_micros": slo_micros
    })
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)
    logging.info(r.text)
    if r.status_code == requests.codes.ok:
        return True
    else:
        logging.warning("Received error status code: {code}".format(code=r.status_code))
        return False


def deploy_model(cm, name, version, input_type, labels=None, image, repo="dockerhub"):
    pass


def publish_model(cm, name, version, input_type, labels=None):
    url = "http://{host}/admin/add_model".format(host=cm.get_admin_addr())
    req_json = json.dumps({
        "model_name": name,
        "model_version": str(version),
        "labels": labels,
        "input_type": input_type,
    })
    headers = {'Content-type': 'application/json'}
    logging.info(req_json)
    r = requests.post(url, headers=headers, data=req_json)
    if r.status_code == requests.codes.ok:
        return True
    else:
        logging.warn("Error publishing model: %s" % r.text)
        return False

def add_container(cm):
    pass

def get_all_apps(cm, verbose=False):
    """Gets information about all applications registered with Clipper.

    Parameters
    ----------
    verbose : bool
        If set to False, the returned list contains the apps' names.
        If set to True, the list contains application info dictionaries.
        These dictionaries have the same attribute name-value pairs that were
        provided to `register_application`.

    Returns
    -------
    list
        Returns a list of information about all apps registered to Clipper.
        If no apps are registered with Clipper, an empty list is returned.
    """
    url = "http://{host}/admin/get_all_applications".format(host=cm.get_admin_addr())
    req_json = json.dumps({"verbose": verbose})
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)

    if r.status_code == requests.codes.ok:
        return r.json()
    else:
        logging.warn(r.text)
        return None

def get_app_info(cm, name):
    """Gets detailed information about a registered application.

    Parameters
    ----------
    name : str
        The name of the application to look up

    Returns
    -------
    dict
        Returns a dictionary with the specified application's info. This
        will contain the attribute name-value pairs that were provided to
        `register_application`. If no application with name `name` is
        registered with Clipper, None is returned.
    """
    url = "http://{host}/admin/get_application".format(host=cm.get_admin_addr())
    req_json = json.dumps({"name": name})
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)

    if r.status_code == requests.codes.ok:
        app_info = r.json()
        if len(app_info) == 0:
            return None
        return app_info
    else:
        logging.warn(r.text)
        return None

def get_all_models(cm, verbose=False):
    """Gets information about all models registered with Clipper.

    Parameters
    ----------
    verbose : bool
        If set to False, the returned list contains the models' names.
        If set to True, the list contains model info dictionaries.

    Returns
    -------
    list
        Returns a list of information about all apps registered to Clipper.
        If no models are registered with Clipper, an empty list is returned.
    """
    url = "http://{host}/admin/get_all_models".format(host=cm.get_admin_addr())
    req_json = json.dumps({"verbose": verbose})
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)

    if r.status_code == requests.codes.ok:
        return r.json()
    else:
        logging.warning(r.text)
        return None

def get_model_info(cm, name, version):
    """Gets detailed information about a registered model.

    Parameters
    ----------
    model_name : str
        The name of the model to look up
    model_version : int
        The version of the model to look up

    Returns
    -------
    dict
        Returns a dictionary with the specified model's info.
        If no model with name `model_name@model_version` is
        registered with Clipper, None is returned.
    """
    url = "http://{host}/admin/get_model".format(host=cm.get_admin_addr())
    req_json = json.dumps({
        "model_name": model_name,
        "model_version": model_version
    })
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)

    if r.status_code == requests.codes.ok:
        app_info = r.json()
        if len(app_info) == 0:
            return None
        return app_info
    else:
        logging.info(r.text)
        return None

def get_all_model_replicas(cm, verbose=False):
    """Gets information about all model containers registered with Clipper.

    Parameters
    ----------
    verbose : bool
        If set to False, the returned list contains the apps' names.
        If set to True, the list contains container info dictionaries.

    Returns
    -------
    list
        Returns a list of information about all model containers known to Clipper.
        If no containers are registered with Clipper, an empty list is returned.
    """
    url = "http://{host}/admin/get_all_containers".format(host=cm.get_admin_addr())
    req_json = json.dumps({"verbose": verbose})
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)
    if r.status_code == requests.codes.ok:
        return r.json()
    else:
        logging.info(r.text)
        return None


def get_model_replica_info(cm, name, version, replica_id):
    """Gets detailed information about a registered container.

    Parameters
    ----------
    name : str
        The name of the container to look up
    version : int
        The version of the container to look up
    replica_id : int
        The container replica to look up

    Returns
    -------
    dict
        A dictionary with the specified container's info.
        If no corresponding container is registered with Clipper, None is returned.
    """
    url = "http://{host}/admin/get_container".format(host=cm.get_admin_addr())
    req_json = json.dumps({
        "model_name": name,
        "model_version": version,
        "replica_id": replica_id,
    })
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)

    if r.status_code == requests.codes.ok:
        app_info = r.json()
        if len(app_info) == 0:
            return None
        return app_info
    else:
        logging.info(r.text)
        return None

def get_clipper_logs(cm):
    pass

def inspect_instance(cm):
    """Fetches metrics from the running Clipper instance.

    Returns
    -------
    str
        The JSON string containing the current set of metrics
        for this instance. On error, the string will be an error message
        (not JSON formatted).
    """
    url = "http://{host}/metrics".format(host=cm.get_query_addr())
    r = requests.get(url)
    try:
        s = r.json()
    except TypeError:
        s = r.text
    return s

def set_model_version(cm, model_name, model_version, num_containers=0):
    """Changes the current model version to `model_version`.

    This method can be used to do model rollback and rollforward to
    any previously deployed version of the model. Note that model
    versions automatically get updated when `deploy_model()` is
    called, so there is no need to manually update the version as well.

    Parameters
    ----------
    model_name : str
        The name of the model
    model_version : int
        The version of the model. Note that `model_version`
        must be a model version that has already been deployed.
    num_containers : int
        The number of new containers to start with the newly
        selected model version.

    """
    # TODO: update to use k8s API
    url = "http://{host}/admin/set_model_version".format(host=cm.get_admin_addr())
    req_json = json.dumps({
        "model_name": model_name,
        "model_version": model_version
    })
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)
    logging.info(r.text)
    for r in range(num_containers):
        add_container(cm, model_name, model_version)

def stop_inactive_models(cm):
    pass

def stop_deployed_models(cm):
    pass

def stop_clipper(cm):
    pass

def stop_all(cm):
    cm.stop_deployed_models()
    cm.stop_clipper()







    
