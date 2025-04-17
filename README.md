## 🤖 PlanAgents：Intelligent Multi-Agent System
PlanAgents is a LangChain-based multi-agent framework leveraging LLMs for plan-execute reasoning. It integrates SQL, RAG, Python, and external APIs, enabling dynamic task planning and tool-augmented execution.
一个基于多智能体框架的智能问答系统，支持复杂任务分解执行、工具调用、多轮对话记忆等功能。


## 🧠 项目亮点

- ✨ 自动规划：复杂问题自动拆解为多个子任务
- 👥 多智能体协作：不同 Agent 执行各自子任务并组合输出
- 🔧 工具集成：本地数据库 + 多 API 工具调用
- 🗣️ 多轮对话 & 记忆：上下文感知，保持连续对话能力

---

## 🔍 功能展示 Function Walkthrough

---

### 🧩 功能一：复杂问题规划 + 工具调用(本地数据库)

**📍示例问题：**
> “帮我查找一下最近购买次数最多的客户最近一次购买商品当天的所有订单中销售最多的商品是哪一件。”

**🧠 Plan：**

1.	Count the number of orders placed by each customer and sort them to identify the customer with the highest purchase frequency.
2.	Retrieve the most recent purchase date from that customer’s order history.
3.	Filter all platform-wide orders that were placed on that specific date.
4.	Aggregate and rank the products sold on that date based on sales volume.
5.	Identify the top-selling product of the day (specify how ties are handled, if applicable).
6.	Given the above steps taken, please respond to the user’s original question.

1. 确定购买次数最多的客户: 统计每个客户的订单数量并排序，选择订单量最大的客户。
2. 获取该客户最近一次购买的日期: 从该客户的订单记录中提取最新订单日期
3. 筛选该日期内所有订单: 包括平台当日所有用户的订单数据
4. 统计当日商品销量: 对当日所有订单中的商品按销售数量汇总排序
5. 输出销量最高的商品: 选择销量最高的商品（处理并列情况需说明）
6. 在完成以上步骤后，请回答用户最初的问题。

**🤖 Agent 规划与执行过程：**

![Step 1 - Planner 分解任务](images/fn1_1.png)  
![Step 2 - 多 Agent 协作](images/fn1_2.png)
![Step 3 - 输出最终回答](images/func1_3.png)

**✅ 最终回答：**

> “根据分析，2025年4月2日销量最高的商品是Comb（梳子），共售出2件。具体分析步骤如下：
1. 确定购买次数最多的客户后，发现其最近一次下单时间为4月2日；
2. 统计该日所有订单发现共有3笔交易；
3. 经商品销量汇总，梳子以2件销量位居榜首，其余商品均为1件。
因此不存在并列情况，梳子为当日最畅销商品。”

---

### 🔧 功能二：工具调用（多 API）

**📍示例问题：**
> “帮我查找今天深圳的天气怎么样，近期有什么重大新闻事件”

**🧠 系统规划 Plan：**

1. Access a reliable weather API or website to retrieve the current weather conditions and forecast for Shenzhen today.
2. Scrape or check credible news sources (e.g., official news portals, reputable media outlets) for the latest major news events related to Shenzhen or nationally, focusing on timeliness and relevance.
3. Summarize the weather data and highlight key news updates concisely.
4. Verify the accuracy of the gathered information by cross-referencing multiple sources if necessary.
5. Given the above steps taken, please respond to the users original question.

1.	访问可靠的天气 API 或网站，获取深圳今天的当前天气状况和天气预报。
2.	从可信的新闻来源（如官方新闻门户网站、知名媒体）抓取或查看与深圳或全国相关的最新重大新闻事件，重点关注时效性和相关性。
3.	简要总结天气数据，并突出重点新闻内容。
4.	如有必要，通过交叉引用多个来源来核实所收集信息的准确性。
5.	在完成以上步骤后，请回答用户最初的问题。

**🔍 工具调用过程：**

![Step 1 - planer分解问题步骤](images/fn2_1.png)  
![Step 2 - execute执行后总结回答](images/fn2_2.png)

**✅ 最终回答：**

根据最新数据交叉验证结果，深圳当前天气存在官方气象局（多云22°C）与国际气象数据（阴天26.7°C）的差异，建议通过深圳气象局官网或12121热线获取权威数据。

重大新闻包括：
1. 华为鸿蒙系统独立发布（2023.10）
2. 深港司法合作创新（2024.03）
3. AI医疗应用全国领先（2024.04）
4. 2025年重大项目投资3.2万亿元\n请访问深圳政府在线(www.sz.gov.cn)获取实时更新。

---

### 🔁 功能三：多轮对话 & 记忆功能

**📍对话过程举例：**

用户第一轮提问：
> “Which day had the highest sales volume in our company?”/
> "我们公司哪一天的销售量最高？"

第二轮：
> "How many customers made purchases in the year preceding that day?"
> “在那一天之前的一年内，我们共有多少顾客消费过？”



**🧠 Plan & Agent 执行：**

![Step 1 - 第一轮planer分析步骤](images/fn3_1.png)  
![Step 2 - 第一轮执行与回答](images/func3_2.png)  
![Step 3 - 第二轮上下文记忆推理](images/func3_3.png)
![Step 4 - 第二轮上下文记忆执行与回答](images/func3_4.png)

**✅ 多轮总结回答：**

> 

---


⸻

🚀 快速开始

pip install -r requirements.txt
python main.py

默认接入 mysql 本地数据库，
默认已配置api_key环境变量

⸻

📄 License

MIT License

⸻

🙌 联系我

wechat:gh1449611723

---
