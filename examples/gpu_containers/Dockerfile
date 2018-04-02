FROM nvidia/cuda:9.0-cudnn7-runtime

# Set anaconda path
ENV ANACONDA /opt/anaconda
ENV PATH $ANACONDA/bin:$PATH

RUN mkdir -p /model \
      && apt-get update \
      && apt-get install -y wget build-essential libzmq5 libzmq5-dev

RUN wget -P /tmp/ http://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
RUN bash /tmp/Anaconda2-4.4.0-Linux-x86_64.sh -b -p $ANACONDA
RUN rm /tmp/Anaconda2-4.4.0-Linux-x86_64.sh
RUN conda install -y pyzmq && pip install prometheus_client pyyaml

WORKDIR /container

COPY containers/python/__init__.py containers/python/rpc.py /container/

COPY monitoring/metrics_config.yaml /container/

COPY clipper_admin/clipper_admin/python_container_conda_deps.txt /lib/

RUN apt-get update --fix-missing \
      && conda config --set ssl_verify no \
      && conda install -c anaconda cloudpickle=0.5.2 \
      && conda install -y --file /lib/python_container_conda_deps.txt \
      && conda install pytorch torchvision -c pytorch 


COPY containers/python/pytorch_container.py containers/python/pytorch_container_entry.sh /container/

CMD ["/container/pytorch_container_entry.sh"]

ENV CLIPPER_MODEL_PATH=/model
