import os
import sys
import numpy as np
import struct
from datetime import datetime
import zmq
import redis
import hashlib
import itertools

KEY_DELIMITER = ":"

SOCKET_POLLING_TIMEOUT_MILLIS = None # never time out
MESSAGE_TYPE_NEW_INNER_NODE = 0
MESSAGE_TYPE_NEW_SOURCE_NODE = 4
MESSAGE_TYPE_NEW_SINK_NODE = 5
MESSAGE_TYPE_BATCH = 1

NODE_TYPE_SOURCE = "source"
NODE_TYPE_SINK = "sink"
NODE_TYPE_INNER = "inner"

"""
Batch messages are the only message types sent and received between
transformer nodes and Clipper.


Structure of a batch message:
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

    def __repr__(self):
        return self.__str__()

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

    def __gt__(self, other):
        if self.host > other.host:
            return True
        return self.port > other.port

    def __hash__(self):
        return hash((self.host, self.port))


class ObjectStore(object):
    def __init__(self, prefix, local_obj_store, cluster):
        """

        Parameters
        ----------

        prefix : str
            A node-specific prefix to append to object IDs for objects created by this node.
        local_object_store : SocketAddress
            The address of the local Redis instance
        cluster : list(SocketAddress)
            A addresses of any remote Redis object store instances in the cluster
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


    def __init__(self, address, node_name, node_type):
        """

        Parameters
        ----------
        address : SocketAddress
            Address of Clipper service
        node_name : str
            Unique identifier for this node in the DAG
        node_name : str
            One of inner, source, sink
        """
        self.address = address
        self.node_name = node_name
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self.socket = self.context.socket(zmq.DEALER)
        self.clipper_addr_string = "tcp://{0}:{1}".format(self.address.host,
                                                 self.address.port)
        if node_type not in [NODE_TYPE_INNER, NODE_TYPE_SOURCE, NODE_TYPE_SINK]:
            raise TransformerError("Invalid node type: %s provided to ClipperConnection" % node_type)
        self.node_type = node_type
        sys.stdout.flush()
        sys.stderr.flush()
        self.connected = False

    def connect(self):
        self.poller.register(self.socket, zmq.POLLIN)
        self.socket.connect(self.clipper_addr_string)
        self.socket.send("".encode('utf-8'), zmq.SNDMORE)
        if self.node_type == NODE_TYPE_SOURCE:
            self.socket.send(struct.pack("<I", MESSAGE_TYPE_NEW_SOURCE_NODE), zmq.SNDMORE)
        elif self.node_type == NODE_TYPE_SINK:
            self.socket.send(struct.pack("<I", MESSAGE_TYPE_NEW_SINK_NODE), zmq.SNDMORE)
        else:
            self.socket.send(struct.pack("<I", MESSAGE_TYPE_NEW_INNER_NODE), zmq.SNDMORE)
        # self.socket.send_string(self.node_name.encode('utf-8'))
        self.socket.send(self.node_name.encode('utf-8'))
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

    def send_batch(self, response_batch):
        """
            Parameters
            ----------
            response_batch : list(string)
                The list of object IDs as strings. Each ID has the following format
                <obj_store_host>:<obj_store_port>:<obj_id_key>

        """
        # Sinks can't send messages to Clipper
        assert self.node_type != NODE_TYPE_SINK
        send_time_start = datetime.now()
        batch_ids = response_batch
        if not type(batch_ids) == list:
            raise TransformerError("Response batch has incorrect type")
        # if len(outputs) != len(prediction_request.inputs):
        #     raise PredictionError(
        #         "Expected model to return %d outputs, found %d outputs" %
        #         (len(prediction_request.inputs), len(outputs)))
        if not type(batch_ids[0]) == str:
            raise PredictionError("Model must return a list of strs. Found %s"
                                % type(batch_ids[0]))
        # addr_string = "%s:%d" % (addr.host, addr.port)
        response_msg = TransformerMessage(batch_ids)
        response_msg.send(self.socket)
        send_time_end = datetime.now()
        print("sent batch of %d ids in %f microseconds" % (len(batch_ids), (send_time_end-send_time_start).microseconds))
        sys.stdout.flush()
        sys.stderr.flush()

    def receive_batch(self):
        """
            Returns
            -------
            list(string) : The next batch of inputs
        """
        # Sources can't receive messages from Clipper
        assert self.node_type != NODE_TYPE_SOURCE
        receivable_sockets = dict(
            self.poller.poll(SOCKET_POLLING_TIMEOUT_MILLIS))
        if self.socket not in receivable_sockets or receivable_sockets[self.socket] != zmq.POLLIN:
            return None

        recv_time_start = datetime.now()
        self.socket.recv()
        msg_type_bytes = self.socket.recv()
        msg_type = struct.unpack("<I", msg_type_bytes)[0]
        if msg_type != MESSAGE_TYPE_BATCH:
            print("Received incorrect message type: %d. Shutting down!" % msg_type)
            self.disconnect()
            raise TransformerException("Received invalid message from Clipper Service")


        batch_size_bytes = self.socket.recv()
        batch_size = struct.unpack("<I", batch_size_bytes)[0]

        # address = self.socket.recv().decode('utf-8')
        batch_ids = [self.socket.recv().decode('utf-8') for _ in range(batch_size)]

        recv_time_end = datetime.now()

        print("received batch of %d ids in %f microseconds" % (batch_size, (recv_time_end-recv_time_start).microseconds))
        sys.stdout.flush()
        sys.stderr.flush()
        return batch_ids


class TransformerMessage():
    def __init__(self, batch_ids):
        """
        Parameters
        ----------
        batch_ids : list(str)
        """
        self.batch_ids = batch_ids


    def send(self, socket):
        """
        Structure of message (each line is separate ZMQ msg)
            4 bytes: message type as int
            4 bytes: size of batch as int
            batch_id_string_0
            batch_id_string_1
            ...
            batch_id_string_k
        """
        socket.send("".encode('utf-8'), flags=zmq.SNDMORE)
        socket.send(struct.pack("<I", MESSAGE_TYPE_BATCH), zmq.SNDMORE)
        socket.send(struct.pack("<I", len(self.batch_ids)), zmq.SNDMORE)
        for i in range(len(self.batch_ids) - 1):
            socket.send(self.batch_ids[i].encode('utf-8'), flags=zmq.SNDMORE)
        socket.send(self.batch_ids[-1].encode('utf-8'))



class Transformer(object):

    def __init__(self, name, transform_func, clipper_addr, local_obj_store, cluster):
        self.name = name
        self.clipper_conn = ClipperConnection(clipper_addr, name, NODE_TYPE_INNER)
        self.clipper_conn.connect()
        self.object_store = ObjectStore(name, local_obj_store, cluster)
        self.transform_func = transform_func

    def run(self):
        while True:
            ids = self.clipper_conn.receive_batch()
            if ids is None:
                continue
            keyfunc = lambda item: item[0]
            addrs_and_ids = sorted([extract_addr(i) for i in ids], key=keyfunc)
            print(addrs_and_ids)
            tuples = []
            for k, g in itertools.groupby(addrs_and_ids, key=keyfunc):
                tuples.extend(self.object_store.get(k, [item[1] for item in g]))
            # tuples = self.object_store.get(addr, ids)
            transformed_tuples = self.transform_func(tuples)
            put_addr, put_objects = self.object_store.put(transformed_tuples)
            completed_batch = [prepend_addr(put_addr, k) for k in put_objects.keys()]
            self.clipper_conn.send_batch(completed_batch)

class Source(object):

    def __init__(self, name, clipper_addr, local_obj_store, cluster):
        self.name = name
        self.clipper_conn = ClipperConnection(clipper_addr, name, NODE_TYPE_SOURCE)
        self.object_store = ObjectStore(name, local_obj_store, cluster)
        self.clipper_conn.connect()
        self.current_tuple_id = 0

    def send(self, items):
        def create_tuple(item):
            t = Tuple(self.current_tuple_id, item)
            self.current_tuple_id += 1
            return t

        item_tuples = [create_tuple(item) for item in items]
        put_addr, put_objects = self.object_store.put(item_tuples)
        batch = [prepend_addr(put_addr, k) for k in put_objects.keys()]
        self.clipper_conn.send_batch(batch)

class Sink(object):

    def __init__(self, name, clipper_addr, local_obj_store, cluster):
        self.name = name
        self.clipper_conn = ClipperConnection(clipper_addr, name, NODE_TYPE_SINK)
        self.clipper_conn.connect()
        self.object_store = ObjectStore(name, local_obj_store, cluster)

    def run(self):
        # completed_batch = None
        while True:
            # Combine notification of completed batch and request for next batch
            # into single message
            ids = self.clipper_conn.receive_batch()
            if ids is None:
                continue
            keyfunc = lambda item: item[0]
            addrs_and_ids = sorted([extract_addr(i) for i in ids], key=keyfunc)
            tuples = []
            for k, g in itertools.groupby(addrs_and_ids, key=keyfunc):
                tuples.extend(self.object_store.get(k, [item[1] for item in g]))
            for t in tuples:
                print(t)




def prepend_addr(addr, id_str):
    full_id = addr.host + KEY_DELIMITER + str(addr.port) + KEY_DELIMITER + id_str
    print(full_id)
    return full_id


def extract_addr(id_str):
    splits = id_str.split(KEY_DELIMITER)
    assert len(splits) == 5
    host = splits[0]
    port = int(splits[1])
    obj_id = KEY_DELIMITER.join(splits[2:])
    address = SocketAddress(host, port)
    # print(address, obj_id)
    return (address, obj_id)
