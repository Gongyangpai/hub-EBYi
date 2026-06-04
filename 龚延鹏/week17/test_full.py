#!/usr/bin/env python
"""完整功能测试脚本 - 测试 enterprise_ai_cache 所有模块"""

import numpy as np
import time

# ==================== 1. 统一向量数据管理 ====================
print("=" * 60)
print("1. 统一向量数据管理测试")
print("=" * 60)

from enterprise_ai_cache.vector import VectorIndexManager, HybridSearchEngine

index_manager = VectorIndexManager()
index_name = "test_comprehensive_idx"

# 清理旧索引
try:
    index_manager.delete_index(index_name)
    print(f"  [清理] 删除旧索引: {index_name}")
except:
    pass

# 创建索引
print(f"  [创建] 索引: {index_name}")
index_manager.create_index(
    index_name=index_name,
    dimensions=768,
    metric="COSINE",
    index_type="HNSW"
)
print(f"  [成功] 索引创建成功")

# 检查索引是否存在
exists = index_manager.index_exists(index_name)
print(f"  [检查] 索引存在: {exists}")

# 列出所有索引
indexes = index_manager.list_indexes()
print(f"  [列表] 所有索引: {indexes}")

# ==================== 2. 混合查询引擎 ====================
print("\n" + "=" * 60)
print("2. 混合查询引擎测试")
print("=" * 60)

engine = HybridSearchEngine(index_name=index_name)

# 2.1 添加向量数据
print("  [添加] 添加测试向量数据...")
vec_fixed = np.random.rand(768).astype(np.float32)
engine.add(id="vec1", vector=vec_fixed, metadata={"user": "a", "category": "test"})
engine.add(id="vec2", vector=np.random.rand(768).astype(np.float32), metadata={"user": "b", "category": "test"})
engine.add(id="vec3", vector=np.random.rand(768).astype(np.float32), metadata={"user": "a", "category": "prod"})
print("  [成功] 向量添加完成")

# 2.2 向量检索
print("  [检索] 向量相似性搜索...")
res = engine.search(query_vector=vec_fixed, top_k=3)
print(f"  [成功] 检索到 {len(res)} 条结果")

# 2.3 带过滤的混合查询
print("  [检索] 向量搜索 + 元数据过滤...")
res = engine.search(
    query_vector=vec_fixed,
    vector_filter={"user": "a"},
    top_k=10
)
print(f"  [成功] 混合查询(过滤)检索到 {len(res)} 条结果")

# 2.4 向量搜索 + 过滤 + 全文搜索
print("  [检索] 向量搜索 + 元数据过滤 + 全文搜索...")
res = engine.search(
    query_vector=vec_fixed,
    vector_filter={"user": "a"},
    text_query="test"
)
print(f"  [成功] 混合查询(过滤+全文)检索到 {len(res)} 条结果")

# 2.5 纯全文搜索
print("  [检索] 纯全文搜索...")
res = engine.text_search(query="test", top_k=10)
print(f"  [成功] 全文搜索检索到 {len(res)} 条结果")

# 2.6 删除向量
print("  [删除] 删除测试向量...")
engine.delete(id="vec1")
print("  [成功] 删除完成")

# ==================== 3. 智能缓存体系 ====================
print("\n" + "=" * 60)
print("3. 智能缓存体系测试")
print("=" * 60)

# 3.1 语义缓存
print("\n  3.1 语义缓存 (SemanticCache)")
from enterprise_ai_cache.cache import SemanticCache
from enterprise_ai_cache.utils import EmbeddingUtility

util = EmbeddingUtility()

# 🔥 修复：清空旧缓存和索引（因为维度从 768 变为 512）
cache_name = "test_semantic_cache"
try:
    import redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    # 删除旧索引
    index_name = f"{cache_name}:vec_idx"
    try:
        r.ft(index_name).dropindex(delete_documents=True)
        print(f"  [清理] 已删除旧索引: {index_name}")
    except:
        pass
    # 删除旧数据
    cursor = 0
    deleted_count = 0
    while True:
        cursor, keys = r.scan(cursor, match=f"{cache_name}:prompts:*", count=100)
        if keys:
            r.delete(*keys)
            deleted_count += len(keys)
        if cursor == 0:
            break
    if deleted_count > 0:
        print(f"  [清理] 已删除 {deleted_count} 条旧数据")
except Exception as e:
    print(f"  [提示] 清理跳过: {e}")

cache = SemanticCache(
    name=cache_name,
    embedding_util=util,
    distance_threshold=1.0
)

# 存储
cache.store(prompt="你好,今天天气怎么样?", response="今天天气很好,适合出门!")
print("  [存储] 语义缓存 - 存入 prompt-response 对")

# 检查/命中
result = cache.check(prompt="你好,今天天气如何?")
print(f"  [检查] 语义缓存 - 命中: {result is not None}")

# 精确获取
cached_resp = cache.get(prompt="你好,今天天气怎么样?")
print(f"  [精确] 精确获取响应: {cached_resp is not None}")

# 统计
stats = cache.stats()
print(f"  [统计] 缓存统计: count={stats['count']}, threshold={stats['distance_threshold']}")


# 3.2 嵌入缓存
print("\n  3.2 嵌入缓存 (EmbeddingsCache)")
from enterprise_ai_cache.cache import EmbeddingsCache

ec = EmbeddingsCache(name="test_embed_cache")

test_text = "这是一段测试文本"
test_vector = np.random.rand(768).astype(np.float32)

# 存储
ec.store(test_text, test_vector)
print(f"  [存储] 嵌入缓存 - 存入: '{test_text[:20]}...'")

# 获取
vec = ec.get(test_text)
print(f"  [获取] 嵌入缓存 - 命中: {vec is not None}")

# 检查存在
exists = ec.exists(test_text)
print(f"  [检查] 存在检查: {exists}")

# 批量操作
batch_texts = ["文本1", "文本2", "文本3"]
batch_vectors = [np.random.rand(768).astype(np.float32) for _ in batch_texts]
ec.store(batch_texts, batch_vectors)
print(f"  [批量] 批量存储 {len(batch_texts)} 条嵌入")

# 统计
embed_stats = ec.stats()
print(f"  [统计] 嵌入缓存统计: count={embed_stats['count']}")

# ==================== 4. 对话历史管理 ====================
print("\n" + "=" * 60)
print("4. 对话历史管理测试 (SemanticMessageHistory)")
print("=" * 60)

from enterprise_ai_cache.cache import SemanticMessageHistory, MessageRole

history = SemanticMessageHistory(name="test_session")

# 添加单条消息
history.add_message({"role": "user", "content": "你好,我想了解一下天气"})
print("  [添加] 单条消息")

# 添加多条消息
history.add_message([
    {"role": "assistant", "content": "今天天气晴朗,温度25度"},
    {"role": "user", "content": "那明天呢?"},
    {"role": "assistant", "content": "明天可能有雨,记得带伞"}
])
print("  [添加] 批量消息")

# 获取最近消息
recent = history.get_recent(top_k=5)
print(f"  [获取] 最近消息数: {len(recent)}")

# 按角色获取
user_msgs = history.get_by_role("user")
print(f"  [过滤] 用户消息数: {len(user_msgs)}")

# 语义搜索
relevant = history.get_relevant("天气", top_k=5)
print(f"  [搜索] 相关消息数: {len(relevant)}")

# 统计
history_stats = history.stats()
print(f"  [统计] 历史统计: total={history_stats['total_messages']}, roles={history_stats['roles']}")

# 导出/导入
exported = history.export()
print(f"  [导出] 导出JSON长度: {len(exported)} 字符")

# ==================== 5. 语义路由 ====================
print("\n" + "=" * 60)
print("5. 语义路由测试 (SemanticRouter)")
print("=" * 60)

from enterprise_ai_cache.router import SemanticRouter

router = SemanticRouter(name="test_router")

# 修复：正确的 add_route 用法
router.add_route(
    references=["你好", "您好", "早上好", "hi", "hello"],
    target="greeting",
    metadata={"type": "问候"},
    distance_threshold=1.0
)
router.add_route(
    references=["天气", "温度", "气候", "weather"],
    target="weather",
    metadata={"type": "天气查询"},
    distance_threshold=1.0
)
router.add_route(
    references=["再见", "拜拜", "bye", "goodbye"],
    target="farewell",
    metadata={"type": "告别"},
    distance_threshold=1.0
)


print("  [添加] 已添加3个路由: greeting, weather, farewell")

router.build()

# 路由匹配
result1 = router.route("你好啊")
print(f"  [路由] '你好啊' -> {result1}")

result2 = router.route("今天温度多少")
print(f"  [路由] '今天温度多少' -> {result2}")

result3 = router.route("再见")
print(f"  [路由] '再见' -> {result3}")

# 列出所有路由
routes = router.list_routes()
print(f"  [列表] 所有路由: {routes}")

# 可调用形式
result4 = router("明天天气如何")
print(f"  [调用] router('明天天气如何') -> {result4}")

# ==================== 测试完成 ====================
print("\n" + "=" * 60)
print("所有测试完成!")
print("=" * 60)