import numpy as np
from clipper_zmq_client import Client

if __name__ == "__main__":
	cl = Client("localhost", 4456, 4455)
	def callback(x):
		print("WOW")

	cl.start()
	cl.send_request("app1", np.array(np.random.rand(10), dtype=np.float64), callback)
	cl.stop()