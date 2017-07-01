import sys
from transformer import *


if __name__ == "__main__":
    ids = sys.argv[1:]
    local_redis = RedisAddress("127.0.0.1", 6380)
    cluster = [RedisAddress("127.0.0.1", 6379)]
    obj_store = ObjectStore("t2", local_redis, cluster)


    fetched_tuples = obj_store.get(cluster[0], ids)
    batch = []
    for f in fetched_tuples:
        # print("ID: %d, data: %s" % (int(f.tuple_id), f.data))
        len_str = str(len(f.data))
        batch.append(Tuple(f.tuple_id, len_str))

        # f.data =
    addr, ids = obj_store.put(batch)
    print(" ".join(list(ids.keys())))
