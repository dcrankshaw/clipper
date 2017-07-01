from transformer import *

if __name__ == "__main__":
    local_redis = RedisAddress("127.0.0.1", 6379)
    cluster = [RedisAddress("127.0.0.1", 6380)]
    obj_store = ObjectStore("t1", local_redis, cluster)

    batch = [Tuple(1, u"run"), Tuple(2, u"the"), Tuple(3, u"jewels")]
    addr, ids = obj_store.put(batch)
    print(" ".join(list(ids.keys())))

    # fetched_tuples = obj_store.get(addr, ids)
    # for f in fetched_tuples:
    #     print("ID: %d, data: %s" % (int(f.tuple_id), f.data.decode("utf-8")))
