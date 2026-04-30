# Phase 2 demo questions

A curated list of queries that **fail under Phase 1 (alias-only) but
succeed under Phase 2** (alias → entity-semantic → news-semantic). Use
this as a smoke-test list when validating retrieval changes — every
question below should resolve to at least one ts_code or surface
relevant articles.

Phase 1 returns `未能识别出A股代码或股票名称` for all of these because
none of them contain a literal stock name, ts_code, or 6-digit symbol.
Phase 2 routes by query flavor:

- **entity flavor** (keywords: 龙头 / 板块 / 行业 / 概念 / 赛道 / 个股 / 标的 / 是谁 / 代表 / 排名 / 前十 / 领头 / 主流 / 核心)
  → cosine search over the 5,190-vector entity index, reranked by news prominence.
- **news flavor** (keywords: 新闻 / 消息 / 动态 / 事件 / 进展 / 公告 / 政策 / 监管 / 调控 / 加息 / 降息 / 利好 / 利空 / 影响 / 冲击 / 风波 / 突发 / 曝光 / 爆雷 / 暴涨 / 暴跌 / 涨停 / 跌停 / 解禁 / 增持 / 减持 / 回购 / 分红 / 业绩预告 / 快报 / 走势 / 行情 / 数据 / 报告 / 发布)
  → cosine search over the 1.95 M-vector news index; ts_codes derived from the matched articles.
- **neutral / mixed** → entity index first, news index as fallback.

---

## A. Sector / sub-sector leader queries (entity flavor)

| # | Question | Expected resolution |
|---|---|---|
| 1 | 新能源车板块龙头是谁，业绩对比如何？ | 比亚迪 (002594.SZ) / 北汽蓝谷 / 长城汽车 — comparison table |
| 2 | 锂电池龙头股有哪些？ | 天齐锂业 (002466.SZ) / 亿纬锂能 (300014.SZ) / 盛新锂能 |
| 3 | 光伏龙头股票推荐 | 阳光电源 (300274.SZ) / 晶澳科技 (002459.SZ) / 太阳能 |
| 4 | 白酒板块龙头有哪些？ | 贵州茅台 (600519.SH) / 老白干酒 / 泸州老窖 |
| 5 | 医美龙头是哪家？ | 爱美客 (300896.SZ) / 奥美医疗 |
| 6 | 算力概念股有哪些龙头？ | 浪潮信息 / 中科曙光 / 立讯精密 |
| 7 | 半导体设备龙头股 | 北方华创 / 中微公司 / 拓荆科技 |
| 8 | 互联网券商龙头 | 浙商证券 / 东方财富 |
| 9 | 创新药板块代表公司 | 恒瑞医药 / 百济神州 / 信达生物 |
| 10 | 储能赛道领头羊 | 阳光电源 / 派能科技 / 宁德时代 |
| 11 | 风电整机龙头 | 金风科技 / 明阳智能 |
| 12 | 工业机器人核心标的 | 埃斯顿 / 汇川技术 / 新时达 |
| 13 | 钠离子电池主流公司 | 宁德时代 / 中科海钠相关股 |
| 14 | 智能驾驶板块代表股 | 德赛西威 / 中科创达 / 经纬恒润 |
| 15 | 国产CPU概念龙头 | 海光信息 / 龙芯中科 / 中科曙光 |
| 16 | 数据中心建设龙头股 | 光环新网 / 宝信软件 / 浪潮信息 |
| 17 | 量子计算概念前十 | 国盾量子 / 光迅科技 / 中际旭创 |
| 18 | 折叠屏手机产业链核心 | 凯盛科技 / 长信科技 / 蓝特光学 |
| 19 | 钙钛矿电池标的 | 宝馨科技 / 协鑫集成 / 京山轻机 |
| 20 | 中药龙头是哪几家 | 片仔癀 / 云南白药 / 同仁堂 |
| 21 | 卫星互联网核心赛道公司 | 中国卫星 / 中国卫通 / 信科移动 |
| 22 | 检测服务行业龙头 | 华测检测 / 广电计量 / 苏试试验 |
| 23 | 海上风电产业链代表股 | 东方电缆 / 大金重工 / 海力风电 |
| 24 | 模拟芯片国产替代领头羊 | 圣邦股份 / 思瑞浦 / 纳芯微 |
| 25 | 减速器主流厂商 | 绿的谐波 / 双环传动 / 中大力德 |

## B. Macro / policy / event queries (news flavor)

These rarely return a clean ts_code — the answer lives in articles.
The retriever returns `ts_codes=[]` and Qwen synthesizes from snippets.

| # | Question | Expected behaviour |
|---|---|---|
| 26 | 美联储加息对A股的影响 | 5 articles: 证券日报 / 川财证券 / 上海证券报 commentary on FOMC ↔ A股 |
| 27 | 降准对银行板块的影响 | RRR-cut articles, banking-sector commentary |
| 28 | 北向资金近期流入流出情况 | northbound flow tracker articles |
| 29 | 房地产调控政策最新动态 | property policy news from major outlets |
| 30 | 注册制改革进展 | registration-system articles, regulatory commentary |
| 31 | 中美贸易摩擦最新消息 | trade-friction headlines + analyst views |
| 32 | 人民币汇率走势分析 | FX articles, central-parity commentary |
| 33 | A股市场退市新规解读 | delisting-rule articles |
| 34 | 央行公开市场操作动态 | OMO / liquidity articles |
| 35 | 雪球结构化产品风险 | structured-product risk articles |
| 36 | 上市公司减持新规影响 | shareholder-reduction rule articles |
| 37 | 转融通业务最新政策 | securities-lending policy articles |
| 38 | 科创板做市商制度进展 | STAR market market-maker articles |
| 39 | 北交所最新IPO情况 | BSE IPO pipeline articles |
| 40 | A股年报披露季关注点 | annual-report season analyst articles |

## C. Concept / theme / trend queries (mixed → entity first)

These have both entity and news keywords; the retriever defaults to
entity first. Expect at least one industry-leader ts_code.

| # | Question | Expected resolution |
|---|---|---|
| 41 | 光伏行业最近的政策利好 | 阳光电源 / 晶澳科技 + recent policy articles |
| 42 | 芯片国产替代最新进展 | 中芯国际 / 沪硅产业 / 雅克科技 + progress articles |
| 43 | 人工智能算力需求驱动哪些公司 | 浪潮信息 / 立讯精密 / 中科曙光 |
| 44 | 新能源车销量数据看哪几家公司 | 比亚迪 / 长城汽车 / 上汽集团 |
| 45 | 5G建设进入哪个阶段，相关公司表现如何 | 中兴通讯 / 烽火通信 / 中国联通 |
| 46 | 创新药出海有哪些代表案例 | 百济神州 / 君实生物 / 信达生物 |
| 47 | 跨境电商行业头部公司 | 安克创新 / 跨境通 / 华凯易佰 |
| 48 | 充电桩行业发展现状 | 特锐德 / 万马股份 / 国电南瑞 |
| 49 | 半导体材料国产化代表企业 | 沪硅产业 / 雅克科技 / 鼎龙股份 |
| 50 | 机器人产业近期趋势及核心标的 | 埃斯顿 / 绿的谐波 / 汇川技术 |

---

## How to use

```bash
# Single-shot smoke test of one question against the running API
./venv/Scripts/python -c "
import requests
r = requests.post('http://127.0.0.1:8080/ask',
                  json={'query':'锂电池龙头股有哪些？','top_k':5},
                  timeout=180)
d = r.json()
print('ts_codes:', d['ts_codes'])
print('articles:', d['n_articles'])
print(d['answer'])"
```

Or load this list into the Gradio UI and iterate. The `/ask` response
includes `ts_codes`, `n_articles`, and `elapsed_seconds` so you can
spot-check both retrieval quality and latency per question.

Each question is intentionally adversarial against Phase 1 (alias-only)
— if any of these *do* resolve under Phase 1, it's because the alias
dictionary picked up an unintended substring match (a regression to
investigate, not a feature).
