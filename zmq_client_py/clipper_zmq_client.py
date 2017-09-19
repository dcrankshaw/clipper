import zmq
import numpy as np
import struct
import time
from datetime import datetime
import socket
import sys

from threading import Lock, Thread
from Queue import Queue

DATA_TYPE_BYTES = 0
DATA_TYPE_INTS = 1
DATA_TYPE_FLOATS = 2
DATA_TYPE_DOUBLES = 3
DATA_TYPE_STRINGS = 4

NUM_REQUESTS_SEND = 10
NUM_RESPONSES_RECV = 10

class Client:

	active = False

	def __init__(self, clipper_host, clipper_send_port, clipper_recv_port):
		self.clipper_host = clipper_host
		self.send_port = clipper_send_port
		self.recv_port = clipper_recv_port
		self.request_queue = Queue()
		self.recv_count = 0

	def start(self):
		global active
		active = True
		self.thread = Thread(target=self._run, args=[])
		self.start_time = datetime.now()
		self.thread.start()

	def stop(self):
		global active
		if active:
			active = False
			self.thread.join()

	def send_request(self, app_name, input_item):
		self.request_queue.put((app_name, input_item))

	def _run(self):
		global active
		clipper_recv_address = "tcp://{0}:{1}".format(self.clipper_host, self.recv_port)
		clipper_send_address = "tcp://{0}:{1}".format(self.clipper_host, self.send_port)
		context = zmq.Context(2)
		recv_socket = context.socket(zmq.DEALER)
		send_socket = context.socket(zmq.DEALER)

		poller = zmq.Poller()
		poller.register(recv_socket, zmq.POLLIN)
		poller.register(send_socket, zmq.POLLIN)

		recv_socket.connect(clipper_recv_address)
		send_socket.connect(clipper_send_address)
		while active:
			if self.request_queue.empty():
				receivable_sockets = dict(poller.poll(1000))
				if socket in receivable_sockets and receivable_sockets[socket] == zmq.POLLIN:
					self._receive_response(socket)
					for i in range(NUM_REQUESTS_SEND - 1):
						receivable_sockets = dict(poller.poll(0))
						if socket in receivable_sockets and receivable_sockets[socket] == zmq.POLLIN:
							self._receive_response(socket)
			else:
				for i in range(NUM_RESPONSES_RECV):
					receivable_sockets = dict(poller.poll(1000))
					if socket in receivable_sockets and receivable_sockets[socket] == zmq.POLLIN:
						self._receive_response(socket)

			self._send_requests(socket)

	def _receive_response(self, socket):
		# Receive delimiter between routing identity and content
		socket.recv()
		data_type_bytes = socket.recv()
		output_data = socket.recv()
		self.recv_count += 1
		if self.recv_count % 200 == 0:
			curr_time = datetime.now()
			latency = (curr_time - self.start_time).total_seconds()
			print("Throughput: {} qps\n".format(200.0 / latency))
			self.start_time = curr_time


	def _send_requests(self, socket):
		i = NUM_REQUESTS_SEND
		while (not self.request_queue.empty()) and i > 0:
			app_name, input_item = self.request_queue.get()
			socket.send("", zmq.SNDMORE)
			socket.send_string(app_name, zmq.SNDMORE)
			socket.send(struct.pack("<I", DATA_TYPE_DOUBLES), zmq.SNDMORE)
			socket.send(struct.pack("<I", len(input_item)), zmq.SNDMORE)
			socket.send(input_item)
			i -= 1
