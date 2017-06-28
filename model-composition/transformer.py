# import os
import sys
# import numpy as np
import zmq
import redis
import hashlib

KEY_DELIMITER = ":"


class TransformerException(Exception):
    pass


class ClipperConnection(object):

    def __init__(self, address):
        self.address = address
        self.context = zmq.Context()


class Tuple(object):

    def __init__(self, tuple_id, data):
        self.tuple_id = tuple_id
        self.data = data

    def __str__(self):
        return "id: {id}, data: {data}".format(id=self.tuple_id, data=str(self.data))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash((self.tuple_id, self.data))


class RedisAddress(object):

    def __init__(self, host, port):
        self.host = host
        self.port = port

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash((self.host, self.port))


class ObjectStore(object):

    def __init__(self, prefix, local_obj_store, cluster):
        """

        Parameters
        ----------

        prefix : str
            A node-specific prefix to append to object IDs for objects created by this node.
        local_object_store : string
            The unix socket path to connect to the local Redis instance on
        cluster : list(RedisAddress)
            A list of addresses to connect to

        """

        self.obj_store_connections = {addr: self._connect(addr) for addr in cluster}
        self.local_obj_store_addr = local_obj_store
        self.local_obj_store_conn = self._connect_local(local_obj_store)
        self.prefix = prefix

    def _connect(self, address):
        return redis.Redis(host=address.host, port=address.port, decode_responses=True)

    def _connect_local(self, address):
        # TODO: can't get Redis + Docker + Unix Sockets working yet
        # return Redis(unix_socket_path=socket)
        return redis.Redis(host=address.host, port=address.port, decode_responses=True)

    def generate_key(self, t):
        return KEY_DELIMITER.join([
            str(t.tuple_id),
            self.prefix,
            hashlib.md5(t.data.encode('utf-8')).hexdigest()
        ])

    def put(self, tuples):
        """
        Parameters
        ----------
        tuples : list(Tuple)
            A list of tuples to insert into the object store.

        Returns
        -------
        A pair. The first item is the address of the object store the objects
        were inserted in to. This will always be the address of the local object store.
        The second item is a list of object store IDs that can be used to look up the objects.
        """
        # TODO: batch these puts into a single request?
        ids = {self.generate_key(t): t for t in tuples}
        pipe = self.local_obj_store_conn.pipeline()
        for key, t in ids.items():
            pipe.set(key, t.data)
        pipe.execute()
        return (self.local_obj_store_addr, ids)

    def get(self, address, ids):
        try:
            if address == self.local_obj_store_addr:
                conn = self.local_obj_store_conn
            else:
                conn = self.obj_store_connections[address]
            pipe = conn.pipeline()
            for key in ids:
                pipe.get(key)
            results = pipe.execute()
            tuples = []
            for (key, val) in zip(ids, results):
                tuple_id = key.split(KEY_DELIMITER)[0]
                assert val is not None
                tuples.append(Tuple(tuple_id, val))
            return tuples

        except KeyError as e:
            msg = "No connection to object store: %s" % e
            print(msg, file=sys.stderr)
            raise TransformerException(msg)


class Transformer(object):

    def __init__(self, name, transform_func, clipper_addr, *args):
        self.clipper_conn = ClipperConnection(clipper_addr)
        self.object_store = ObjectStore(name, args)
        self.transform_func = transform_func

    def run(self):
        completed_ids = None
        while True:
            # Combine notification of completed batch and request for next batch
            # into single message
            ids = self.clipper_conn.get_next_batch(completed_batch_ids=completed_ids)
            tuples = self.object_store.fetch(ids)
            transformed_tuples = self.transform_func(tuples)
            completed_ids = self.object_store.put(transformed_tuples)



if __name__ == "__main__":
    local_redis = RedisAddress("127.0.0.1", 6379)
    cluster = [RedisAddress("127.0.0.1", 6380)]
    obj_store = ObjectStore("test", local_redis, cluster)

    batch = [Tuple(1, u"run"), Tuple(2, u"the"), Tuple(3, u"jewels")]
    addr, ids = obj_store.put(batch)

    fetched_tuples = obj_store.get(addr, ids)
    for f in fetched_tuples:
        print("ID: %d, data: %s" % (int(f.tuple_id), f.data.decode("utf-8")))

    # print([str(f) for f in fetched_tuples])






