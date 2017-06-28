import os
import sys
import numpy as np
import struct
from datetime import datetime
import zmq
import redis
import hashlib

KEY_DELIMITER = ":"

SOCKET_POLLING_TIMEOUT_MILLIS = 0 # never time out
MESSAGE_TYPE_BATCH = 1
MESSAGE_TYPE_NEW_CONTAINER = 0

"""
Batch messages are the only message types sent and received between
transformer nodes and Clipper.


Structure of a batch message:
    First item: Address of object store containing the IDS.
    List of new object store IDs.
"""


class TransformerException(Exception):
    pass



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


class SocketAddress(object):

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
        cluster : list(SocketAddress)
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
        The second item is a dict of <object store ID, Tuple> pairs. The object store IDs
        are strings that can be used by other nodes to look up the objects.
        """
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

class ClipperConnection(object):

    def __init__(self, address, node_name):
        """

        Parameters
        ----------
        address : SocketAddress
            Address of Clipper service
        node_name : str
            Unique identifier for this node in the DAG
        """
        self.address = address
        self.node_name = node_name
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self.socket = self.context.socket(zmq.DEALER)
        self.clipper_addr_string = "tcp://{0}:{1}".format(self.address.host,
                                                 self.address.port)
        sys.stdout.flush()
        sys.stderr.flush()
        self.connected = False

    def connect(self):
        self.poller.register(self.socket, zmq.POLLIN)
        self.socket.connect(self.clipper_addr_string)
        socket.send("", zmq.SNDMORE)
        socket.send(struct.pack("<I", MESSAGE_TYPE_NEW_CONTAINER), zmq.SNDMORE)
        socket.send_string(self.node_name.encode('utf-8'))
        print("Sent container metadata!")
        self.connected = True
        sys.stdout.flush()
        sys.stderr.flush()

    def disconnect(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.connected = False
        self.poller.unregister(self.socket)
        self.socket.close()

    def get_next_batch(response_batch=None):
        """
            Parameters
            ----------
            response_batch : (SocketAddress, list(string)), optional
                A tuple where the first item is the address of the object store that the tuples
                have been inserted into and the second item is the list of object IDs as strings.

            Returns
            -------
            (SocketAddress, list(string)) : The next batch of inputs
        """
        if response_batch is not None:
            send_time_start = datetime.now()
            addr, batch_ids = response_batch
            if not type(batch_ids) == list:
                raise TransformerError("Response batch has incorrect type")
            # if len(outputs) != len(prediction_request.inputs):
            #     raise PredictionError(
            #         "Expected model to return %d outputs, found %d outputs" %
            #         (len(prediction_request.inputs), len(outputs)))
            if not type(outputs[0]) == str:
                raise PredictionError("Model must return a list of strs. Found %s"
                                    % type(outputs[0]))
            addr_string = "%s:%d" % (addr.host, addr.port)
            response_msg = TransformerMessage(addr_string, batch_ids)
            response_msg.send(self.socket)
            send_time_end = datetime.now()
            print("sent batch of %d ids in %f microseconds" % (len(batch_ids), (send_time_end-send_time_start).microseconds))
            sys.stdout.flush()
            sys.stderr.flush()

        receivable_sockets = dict(
            self.poller.poll(SOCKET_POLLING_TIMEOUT_MILLIS))
        if self.socket not in receivable_sockets or receivable_sockets[self.socket] != zmq.POLLIN:
            return None

        recv_time_start = datetime.now()
        self.socket.recv()
        msg_type_bytes = self.socket.recv()
        msg_type = struct.unpack("<I", msg_type_bytes)[0]
        if msg_type != MESSAGE_TYPE_CONTAINER_CONTENT:
            print("Received incorrect message type: %d. Shutting down!" % msg_type)
            self.disconnect()
            raise TransformerException("Received invalid message from Clipper Service")


        batch_size_bytes = self.socket.recv()
        batch_size = struct.unpack("<I", batch_size_bytes)[0]

        address = self.socket.recv().decode('utf-8')
        batch_ids = [self.socket.recv().decode('utf-8') for _ in range(batch_size)]

        recv_time_end = datetime.now()

        print("received batch of %d ids in %f microseconds" % (batch_size, (recv_time_end-recv_time_start).microseconds))
        sys.stdout.flush()
        sys.stderr.flush()
        host, port = address.split(":")
        return (SocketAddress(host, int(port)), batch_ids)


class TransformerMessage():
    def __init__(self, address, batch_ids):
        """
        Parameters
        ----------
        address : str
        batch_ids : list(str)
        """
        self.address = address
        self.batch_ids = batch_ids


    def send(self, socket):
        """
        Structure of message (each line is separate ZMQ msg)
            4 bytes: message type as int
            4 bytes: size of batch as int
            address
            batch_id_string_0
            batch_id_string_1
            ...
            batch_id_string_k
        """
        socket.send("", flags=zmq.SNDMORE)
        socket.send(int(MESSAGE_TYPE_BATCH), flags=zmq.SNDMORE)
        socket.send(len(self.batch_ids), flags=zmq.SNDMORE)
        socket.send(self.address.encode('utf-8'), flags=zmq.SNDMORE)
        for i in range(len(batch_ids) - 1):
            socket.send(self.batch_ids[i].encode('utf-8'), flags=zmq.SNDMORE)
        socket.send(self.batch_ids[-1].encode('utf-8'))



class Transformer(object):

    def __init__(self, name, transform_func, clipper_addr, *args):
        self.name = name
        self.clipper_conn = ClipperConnection(clipper_addr, name)
        self.object_store = ObjectStore(name, args)
        self.transform_func = transform_func

    def run(self):
        completed_batch = None
        self.clipper_conn.connect()
        while True:
            # Combine notification of completed batch and request for next batch
            # into single message
            addr, ids = self.clipper_conn.get_next_batch(completed_batch_ids=completed_ids)
            tuples = self.object_store.get(addr, ids)
            transformed_tuples = self.transform_func(tuples)
            put_addr, put_objects = self.object_store.put(transformed_tuples)
            put_object_ids = list(put_objects.keys())
            completed_batch = (put_addr, put_object_ids)




if __name__ == "__main__":
    local_redis = SocketAddress("127.0.0.1", 6379)
    cluster = [SocketAddress("127.0.0.1", 6380)]
    obj_store = ObjectStore("test", local_redis, cluster)

    batch = [Tuple(1, u"run"), Tuple(2, u"the"), Tuple(3, u"jewels")]
    addr, ids = obj_store.put(batch)

    fetched_tuples = obj_store.get(addr, ids)
    for f in fetched_tuples:
        print("ID: %d, data: %s" % (int(f.tuple_id), f.data.decode("utf-8")))

    # print([str(f) for f in fetched_tuples])






