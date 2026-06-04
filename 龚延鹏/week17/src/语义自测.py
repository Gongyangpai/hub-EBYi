from enterprise_ai_cache.router import SemanticRouter
router = SemanticRouter(name="test_router")
# 清空历史脏数据
# router.delete()
# 增加更多参考例句，降低匹配难度
router.add_route(references=["你好","哈喽","早上好","你好呀"],target="greeting")
router.add_route(references=["气温多少、今天温度、查天气、明日气温"],target="weather")
router.add_route(references=["拜拜、再见、回见、告辞"],target="farewell")

# 打印原始返回对象，看是真无匹配还是只取name为空
res1 = router.route("你好啊")
print("原始返回对象:",res1,"取值:",res1 if res1 else None)
res2 = router("今天温度多少")
print("原始返回对象:",res2,"取值:",res2 if res2 else None)