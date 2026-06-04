# debug_full_flow.py
import redis
import numpy as np

r = redis.Redis(host='localhost', port=6379, db=0)

print("=" * 80)
print("步骤 1: 检查索引配置")
print("=" * 80)
try:
    info = r.ft('vec:test_comprehensive_idx').info()
    print(f"索引名称: vec:test_comprehensive_idx")
    print(f"文档数: {info.get(b'num_docs', info.get('num_docs', 0))}")

    # 打印 schema 字段
    if b'attributes' in info:
        attrs = info[b'attributes']
        print(f"\nSchema 字段:")
        for attr in attrs:
            if isinstance(attr, dict):
                print(f"  - {attr}")
except Exception as e:
    print(f"索引查询失败: {e}")

print("\n" + "=" * 80)
print("步骤 2: 检查存储的数据")
print("=" * 80)
keys = r.keys('doc:test_comprehensive_idx:*')
print(f"找到 {len(keys)} 个 keys")

for key in keys[:2]:
    print(f"\nKey: {key}")
    data = r.hgetall(key)
    print(f"  字段列表: {list(data.keys())}")

    # 检查每个字段
    for field_name, field_value in data.items():
        fname = field_name.decode() if isinstance(field_name, bytes) else field_name
        if fname == 'vector':
            vec = np.frombuffer(field_value, dtype=np.float32)
            print(f"  {fname}: shape={len(vec)}, norm={np.linalg.norm(vec):.4f}, first_3={vec[:3]}")
        else:
            fval = field_value.decode() if isinstance(field_value, bytes) else field_value
            print(f"  {fname}: {fval[:100] if len(str(fval)) > 100 else fval}")

print("\n" + "=" * 80)
print("步骤 3: 直接使用 execute_command 执行 KNN 搜索")
print("=" * 80)

# 使用第一个向量的值进行搜索
if keys:
    first_key = keys[0]
    data = r.hgetall(first_key)
    stored_vec = np.frombuffer(data[b'vector'], dtype=np.float32)
    print(f"使用向量: norm={np.linalg.norm(stored_vec):.4f}")

    # 执行 KNN 搜索
    result = r.execute_command(
        'FT.SEARCH', 'vec:test_comprehensive_idx',
        '*=>[KNN 5 @vector $vec AS score]',
        'PARAMS', '2', 'vec', stored_vec.tobytes(),
        'RETURN', '4', 'id', 'score', 'metadata', '_text_meta',
        'DIALECT', '2'
    )

    print(f"\n返回类型: {type(result)}")
    if isinstance(result, dict):
        print(f"字典键: {list(result.keys())}")
        print(f"total_results: {result.get(b'total_results', result.get('total_results'))}")

        results_list = result.get(b'results', result.get('results', []))
        print(f"results 数量: {len(results_list)}")

        for idx, item in enumerate(results_list):
            print(f"\n结果 {idx}:")
            print(f"  类型: {type(item)}")
            print(f"  键: {list(item.keys()) if isinstance(item, dict) else 'N/A'}")
            print(f"  内容: {item}")
    elif isinstance(result, list):
        print(f"列表长度: {len(result)}")
        print(f"前 5 个元素: {result[:5]}")

print("\n" + "=" * 80)
print("步骤 4: 测试 HybridSearchEngine")
print("=" * 80)
from enterprise_ai_cache.vector import HybridSearchEngine

engine = HybridSearchEngine(index_name="test_comprehensive_idx")
print(f"count(): {engine.count()}")

result = engine.search(query_vector=stored_vec.tolist())
print(f"search() 返回 {len(result)} 条结果")
for res in result:
    print(f"  - {res}")
