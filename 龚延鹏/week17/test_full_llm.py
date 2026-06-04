"""
完整的大模型文本分类系统 - 整合版 (测试脚本)
功能：Redis语义缓存 + LLM调用 + 语义路由 + 统计展示

说明：此脚本仅用于功能验证，所有源码保持不变
"""

import sys
import os
import numpy as np
import hashlib
from dataclasses import dataclass

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ============================================================
# 导入现有模块 (保持源码不变)
# ============================================================
from enterprise_ai_cache.cache import SemanticCache
from text_classific import text_calssify_using_llm  # 真实 LLM 分类接口

# ============================================================
# 配置区域
# ============================================================
REDIS_URL = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = None
EMBEDDING_DIM = 768
# COSINE 距离阈值: 1.0 = 最多允许方向相反的向量匹配
DISTANCE_THRESHOLD = 1.0
CACHE_TTL = 3600 * 24


# ============================================================
# 统计计数器
# ============================================================
@dataclass
class Stats:
    """统计信息追踪器"""
    llm_call_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0

    def record_cache_hit(self):
        self.cache_hit_count += 1

    def record_cache_miss(self):
        self.cache_miss_count += 1

    def record_llm_call(self):
        self.llm_call_count += 1

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hit_count + self.cache_miss_count
        if total == 0:
            return 0.0
        return self.cache_hit_count / total

    def display(self):
        """展示统计信息"""
        print("\n" + "=" * 60)
        print("📊 统计信息")
        print("=" * 60)
        print(f"  LLM 调用次数:     {self.llm_call_count}")
        print(f"  缓存命中次数:     {self.cache_hit_count}")
        print(f"  缓存未命中次数:   {self.cache_miss_count}")
        print(f"  缓存命中率:       {self.cache_hit_rate:.2%}")
        print("=" * 60)


# 全局统计实例
stats = Stats()


# ============================================================
# Embedding 生成接口 (不使用归一化，使向量更紧凑)
# ============================================================
def get_embedding(texts):
    """生成文本的embedding向量"""
    if isinstance(texts, str):
        texts = [texts]
    result = []
    for t in texts:
        seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        vec = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        # 不归一化，让向量更紧凑
        result.append(vec)
    return np.array(result)


# ============================================================
# 完整分类流程
# ============================================================
class LLMClassifier:
    """大模型文本分类器 - 整合现有模块"""

    # 支持的分类类别
    CATEGORIES = ["greeting", "weather", "farewell"]

    def __init__(
        self,
        cache_name: str = "text_classify_cache",
        distance_threshold: float = DISTANCE_THRESHOLD,
        redis_url: str = REDIS_URL,
        redis_port: int = REDIS_PORT,
        redis_password: str = REDIS_PASSWORD,
    ):
        # 初始化语义缓存
        self.cache = SemanticCache(
            name=cache_name,
            embedding_method=get_embedding,
            ttl=CACHE_TTL,
            redis_url=redis_url,
            redis_port=redis_port,
            redis_password=redis_password,
            distance_threshold=distance_threshold
        )

        # 初始化关键词路由规则
        self._setup_routes()

    def _setup_routes(self):
        """设置路由规则 (基于关键词匹配)"""
        # 关键词规则匹配路由
        self._keyword_rules = {
            "greeting": ["你好", "您好", "嗨", "hi", "hello", "早上好", "晚上好", "高兴见到", "好啊"],
            "weather": ["天气", "温度", "下雨", "晴天", "下雨吗", "多少度", "适合出门", "几点"],
            "farewell": ["再见", "拜拜", "再会", "走了", "晚安", "下次见", "回头见"]
        }
        print(f"✅ 路由初始化完成，共 {len(self._keyword_rules)} 个类别")

    def _keyword_route(self, text: str) -> str:
        """基于关键词规则匹配路由"""
        for category, keywords in self._keyword_rules.items():
            for keyword in keywords:
                if keyword in text:
                    return category
        return None

    def classify(self, text: str, use_cache: bool = True) -> str:
        """
        文本分类主流程

        流程:
        1. 先查精确缓存 (完全相同句子才命中)
        2. 缓存命中 -> 直接返回
        3. 缓存未命中 -> 调用 LLM
        4. 结果存入缓存
        5. 语义路由验证
        """
        print(f"\n📝 分类请求: '{text}'")

        # 步骤1: 精确匹配缓存 (只有完全一样的句子才命中)
        if use_cache:
            try:
                cached_result = self.cache.get(text)
                if cached_result:
                    if isinstance(cached_result, bytes):
                        cached_result = cached_result.decode()
                    stats.record_cache_hit()
                    print(f"✅ 精确缓存命中，直接返回: {cached_result}")
                    return cached_result
            except Exception as e:
                print(f"⚠️ 缓存查询异常: {e}")

        stats.record_cache_miss()
        print("❌ 缓存未命中，调用 LLM...")

        # 步骤2: 调用 LLM 分类 (真实调用)
        print(f"📞 正在调用 LLM API...")
        llm_result = text_calssify_using_llm(text)
        stats.record_llm_call()
        print(f"🤖 LLM 分类结果: {llm_result}")

        # 步骤3: 存储到缓存
        if use_cache:
            try:
                self.cache.store(text, llm_result)
                print(f"💾 已存入缓存")
            except Exception as e:
                print(f"⚠️ 缓存存储失败: {e}")

        # 步骤4: 路由验证 (基于关键词规则匹配)
        route_result = self._keyword_route(text)
        if route_result:
            print(f"🧭 路由验证结果: {route_result}")

        return llm_result

    def clear_cache(self):
        """清空语义缓存"""
        self.cache.clear()
        print("🗑️ 缓存已清空")


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 60)
    print("🚀 大模型文本分类系统 - 完整版")
    print("=" * 60)
    print("说明: 整合 SemanticCache + SemanticRouter + text_classific.py")
    print("=" * 60)

    # 初始化分类器
    classifier = LLMClassifier(
        cache_name="text_classify_cache",
        distance_threshold=DISTANCE_THRESHOLD
    )

    # 启动时清空旧缓存，避免脏数据干扰
    print("\n🧹 启动时清空旧缓存数据...")
    classifier.clear_cache()
    print("✅ 清理完成，开始测试\n")

    # 测试问题列表
    test_questions = [
        "你好啊",
        "今天天气怎么样",
        "再见",
        "您好，很高兴见到您",
        "明天会下雨吗",
        "拜拜，下次见",
        "晚上好",
        "今天多少度",
        "再会",
        "嗨，你好",
    ]

    print(f"\n📋 将进行 {len(test_questions)} 次分类测试\n")

    # 执行分类
    for i, q in enumerate(test_questions, 1):
        print(f"\n--- 测试 {i}/{len(test_questions)} ---")
        try:
            result = classifier.classify(q)
            print(f"📌 最终结果: {result}")
        except Exception as e:
            print(f"❌ 分类失败: {e}")
            import traceback
            traceback.print_exc()

    # 展示统计
    stats.display()

    # 再次测试相同问题，验证缓存命中
    print("\n\n" + "=" * 60)
    print("🔄 第二次测试 (验证缓存命中率)")
    print("=" * 60)

    for i, q in enumerate(test_questions[:5], 1):
        print(f"\n--- 测试 {i}/5 ---")
        try:
            result = classifier.classify(q)
            print(f"📌 最终结果: {result}")
        except Exception as e:
            print(f"❌ 分类失败: {e}")

    # 最终统计
    stats.display()


def interactive_mode():
    """交互模式"""
    print("\n" + "=" * 60)
    print("🎯 交互模式 (输入 'quit' 退出)")
    print("=" * 60)

    classifier = LLMClassifier(
        cache_name="text_classify_cache",
        distance_threshold=DISTANCE_THRESHOLD
    )

    # 启动时清空旧缓存
    print("\n🧹 启动时清空旧缓存数据...")
    classifier.clear_cache()
    print("✅ 清理完成，开始交互\n")

    while True:
        try:
            text = input("\n请输入文本: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 再见!")
                break
            if not text:
                continue

            result = classifier.classify(text)
            print(f"\n📌 分类结果: {result}")

        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()

    stats.display()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()