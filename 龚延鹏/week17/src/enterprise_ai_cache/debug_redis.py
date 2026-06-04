# # debug_redis.py
# import redis
# import numpy as np
#
# r = redis.Redis(host='localhost', port=6379, db=0)
#
# # 1. 检查索引是否存在
# try:
#     info = r.ft('vec:test_comprehensive_idx').info()
#     print(f"索引信息:")
#     print(f"  - num_docs: {info.get('num_docs', 0)}")
#     print(f"  - num_records: {info.get('num_records', 0)}")
# except Exception as e:
#     print(f"索引不存在或查询失败: {e}")
#
# # 2. 检查是否有数据
# keys = r.keys('doc:test_comprehensive_idx:*')
# print(f"\n找到的 keys: {len(keys)}")
# stored_vectors = []
# for key in keys[:3]:
#     print(f"  - {key}")
#     data = r.hgetall(key)
#     print(f"    字段数: {len(data)}")
#     if b'vector' in data:
#         vec = np.frombuffer(data[b'vector'], dtype=np.float32)
#         stored_vectors.append(vec)
#         print(f"    向量维度: {len(vec)}")
#         print(f"    向量范数: {np.linalg.norm(vec):.4f}")
#
# # 3. 尝试简单搜索 - 使用随机向量
# print("\n--- 测试 1: 随机查询向量 ---")
# try:
#     from redis.commands.search.query import Query
#
#     query_vec = np.random.rand(768).astype(np.float32)
#     norm = np.linalg.norm(query_vec)
#     if norm > 0:
#         query_vec = query_vec / norm
#     query_bytes = query_vec.tobytes()
#
#     q = Query("*=>[KNN 5 @vector $vec AS score]").return_fields("id", "score").dialect(2)
#     res = r.ft('vec:test_comprehensive_idx').search(q, query_params={"vec": query_bytes})
#     print(f"KNN 搜索结果: {len(res.docs)} 条")
#     for doc in res.docs:
#         print(f"  - id: {doc.id}, score: {doc.score}")
# except Exception as e:
#     print(f"搜索失败: {e}")
#
# # 4. 尝试使用已存储的向量进行搜索（应该能找到自己）
# if stored_vectors:
#     print("\n--- 测试 2: 使用已存储的向量搜索 ---")
#     try:
#         query_vec = stored_vectors[0]  # 使用第一个存储的向量
#         query_bytes = query_vec.tobytes()
#
#         q = Query("*=>[KNN 5 @vector $vec AS score]").return_fields("id", "score").dialect(2)
#         res = r.ft('vec:test_comprehensive_idx').search(q, query_params={"vec": query_bytes})
#         print(f"KNN 搜索结果: {len(res.docs)} 条")
#         for doc in res.docs:
#             print(f"  - id: {doc.id}, score: {doc.score}")
#     except Exception as e:
#         print(f"搜索失败: {e}")
#
# # 5. 尝试不带过滤的简单搜索
# print("\n--- 测试 3: 最简单的搜索 ---")
# try:
#     query_vec = np.random.rand(768).astype(np.float32)
#     norm = np.linalg.norm(query_vec)
#     if norm > 0:
#         query_vec = query_vec / norm
#     query_bytes = query_vec.tobytes()
#
#     q = Query("*=>[KNN 10 @vector $vec AS score]").dialect(2)
#     res = r.ft('vec:test_comprehensive_idx').search(q, query_params={"vec": query_bytes})
#     print(f"搜索结果: {len(res.docs)} 条")
# except Exception as e:
#     print(f"搜索失败: {e}")


# debug_redis.py
# import redis
# import numpy as np
#
# r = redis.Redis(host='localhost', port=6379, db=0)
#
# # 1. 检查索引详细信息
# print("=== 索引详细信息 ===")
# try:
#     info = r.ft('vec:test_comprehensive_idx').info()
#     print(f"num_docs: {info.get('num_docs', 0)}")
#     print(f"num_records: {info.get('num_records', 0)}")
#
#     # 打印所有索引信息
#     print("\n完整索引信息:")
#     for key, value in info.items():
#         if isinstance(key, bytes):
#             key = key.decode()
#         print(f"  {key}: {value}")
# except Exception as e:
#     print(f"索引不存在或查询失败: {e}")
#     import traceback
#
#     traceback.print_exc()
#
# # 2. 检查数据
# print("\n=== 数据检查 ===")
# keys = r.keys('doc:test_comprehensive_idx:*')
# print(f"找到的 keys: {len(keys)}")
# for key in keys:
#     data = r.hgetall(key)
#     print(f"\nKey: {key}")
#     for k, v in data.items():
#         if isinstance(k, bytes):
#             k = k.decode()
#         if k == 'vector':
#             vec = np.frombuffer(v, dtype=np.float32)
#             print(f"  {k}: shape={len(vec)}, norm={np.linalg.norm(vec):.4f}")
#         else:
#             if isinstance(v, bytes):
#                 v = v.decode()
#             print(f"  {k}: {v[:50] if len(str(v)) > 50 else v}")
#
# # 3. 尝试 FT.SEARCH 命令直接执行
# print("\n=== 直接使用 Redis 命令测试 ===")
# try:
#     # 尝试最简单的搜索
#     result = r.execute_command('FT.SEARCH', 'vec:test_comprehensive_idx', '*', 'LIMIT', '0', '10')
#     print(f"FT.SEARCH * 结果: {result}")
# except Exception as e:
#     print(f"FT.SEARCH 失败: {e}")
#
# # 4. 尝试 KNN 搜索
# print("\n=== KNN 搜索测试 ===")
# try:
#     query_vec = np.random.rand(768).astype(np.float32)
#     norm = np.linalg.norm(query_vec)
#     if norm > 0:
#         query_vec = query_vec / norm
#
#     result = r.execute_command(
#         'FT.SEARCH', 'vec:test_comprehensive_idx',
#         '*=>[KNN 5 @vector $vec AS score]',
#         'PARAMS', '2', 'vec', query_vec.tobytes(),
#         'RETURN', '2', 'id', 'score',
#         'DIALECT', '2'
#     )
#     print(f"KNN 搜索结果: {result}")
# except Exception as e:
#     print(f"KNN 搜索失败: {e}")
#     import traceback
#
#     traceback.print_exc()

# debug_redis.py - 简化版
# import redis
# import numpy as np
#
# r = redis.Redis(host='localhost', port=6379, db=0)
#
# print("=== KNN 搜索测试 ===")
#
# # 获取一个已存储的向量
# keys = r.keys('doc:test_comprehensive_idx:*')
# if keys:
#     data = r.hgetall(keys[0])
#     stored_vec = np.frombuffer(data[b'vector'], dtype=np.float32)
#     print(f"使用存储的向量: norm={np.linalg.norm(stored_vec):.4f}")
#
#     # 方法 1: 使用 execute_command
#     print("\n--- 方法 1: execute_command ---")
#     try:
#         result = r.execute_command(
#             'FT.SEARCH', 'vec:test_comprehensive_idx',
#             '*=>[KNN 5 @vector $vec AS score]',
#             'PARAMS', '2', 'vec', stored_vec.tobytes(),
#             'RETURN', '2', 'id', 'score',
#             'SORTBY', 'score', 'ASC',
#             'DIALECT', '2'
#         )
#         print(f"结果类型: {type(result)}")
#         print(f"结果长度: {len(result) if isinstance(result, (list, dict)) else 'N/A'}")
#         print(f"结果: {result}")
#     except Exception as e:
#         print(f"失败: {e}")
#         import traceback
#
#         traceback.print_exc()
#
#     # 方法 2: 使用 ft().search()
#     print("\n--- 方法 2: ft().search() ---")
#     try:
#         from redis.commands.search.query import Query
#
#         q = Query("*=>[KNN 5 @vector $vec AS score]").return_fields("id", "score").sort_by("score").dialect(2)
#         res = r.ft('vec:test_comprehensive_idx').search(q, query_params={"vec": stored_vec.tobytes()})
#         print(f"找到 {len(res.docs)} 条结果")
#         for doc in res.docs:
#             print(f"  id={doc.id}, score={doc.score}")
#     except Exception as e:
#         print(f"失败: {e}")
#         import traceback
#
#         traceback.print_exc()
# else:
#     print("没有找到数据")

# 快速检查
from enterprise_ai_cache.vector import HybridSearchEngine, VectorIndexManager
import numpy as np

engine = HybridSearchEngine(index_name="test_comprehensive_idx")
print(f"索引中的文档数: {engine.count()}")

# 尝试搜索
result = engine.search(query_vector=np.random.rand(768))
print(f"搜索结果: {len(result)} 条")