from transformer import *
import sys


if __name__ == "__main__":
    ids = sys.argv[1:]
    local_redis = RedisAddress("127.0.0.1", 6379)
    cluster = [RedisAddress("127.0.0.1", 6380)]
    obj_store = ObjectStore("t3", local_redis, cluster)

    fetched_tuples = obj_store.get(cluster[0], ids)
    for f in fetched_tuples:
        print("ID: %d, data: %s" % (int(f.tuple_id), f.data))
