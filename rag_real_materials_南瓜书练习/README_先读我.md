# 用完整中文教材练 RAG：开工说明

## 这是什么
本包只含原版《南瓜书》PDF、来源与许可和本说明。不含完整解答代码。
原文版本 1.9.9（2023.03），196 个 PDF 页面，约 4.05 MB。
它是机器学习公式解析教材，不是一本 RAG 教材。此处把它当作被检索的知识库；无需读懂全书公式。
Datawhale《动手学大模型应用开发》第三章确实使用这份 PDF 演示加载、切分和向量检索。
原课程是含完整实现的实操教程，不是留空作业，也没有自动评分。

## 先打开哪个文件
先打开 pumpkin_book.pdf 的 PDF 第 18 页（纸面第 5 页）：第 2 章“模型评估与选择”。
它有完整目录、章节、公式与参考资料，原文未经删节或改写。

## 原版教程与代码
- 课程：https://github.com/datawhalechina/llm-universe
- 原版Notebook：https://github.com/datawhalechina/llm-universe/blob/main/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93/C3.ipynb
- 评估Notebook：https://github.com/datawhalechina/llm-universe/blob/main/notebook/C5%20%E7%B3%BB%E7%BB%9F%E8%AF%84%E4%BC%B0%E4%B8%8E%E4%BC%98%E5%8C%96/C5.ipynb
- 原始PDF：https://raw.githubusercontent.com/datawhalechina/llm-universe/main/data_base/knowledge_db/pumkin_book/pumpkin_book.pdf
- 许可协议：https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.zh-hans

## 2—3 小时的建议练习（我们补充，不是官方布置）
1. 20 分钟：打开第 18 页并确认问题证据。用 PyMuPDF 逐页提取文本，保留 PDF 页码（从 1 开始），不把页码仅存在最后打印结果里。避免将公式识别质量当作正文识别质量。
2. 40 分钟：逐页切分，约 500 字、重叠 50 字；保留 source、pdf_page、chunk_id。尽量先按段落断开，不跨页拼接。过滤反复出现的页眉页脚，但保留未经清洗的原始文本用于核对。
3. 50 分钟：jieba 中文分词 + rank-bm25 做检索，打印前三个片段、分数、页码。此阶段无需模型 API，不要执行教材里的任何代码。
4. 30 分钟：自写五道题并逐题先手工标注证据页，再算 hit@3：前三个片段中是否有包含正确答案的证据。报告“命中数/5”，不能只凭答案听着对就判成功。加入一道书里不存在的个人信息问题；BM25 总会排序，但排序不是证据存在的保证。
5. 选做：把证据交给现有模型 API 生成回答并引用页码；没有证据就说明未找到。向量检索按原课程后续完成，不要将 BM25 标成完整 RAG 或混合检索。

建议新建虚拟环境，只安装当前任务需要的 pymupdf、jieba、rank-bm25，不要第一步安装整门课程的所有服务依赖。

## 第一题与核对位置
题目：教材列出了哪三种模型评估方法？交叉验证通常用来比较什么？
出处：PDF 第 18 页，纸面第 5 页，2.2“评估方法”。
核对答案：留出法、交叉验证法、自助法；交叉验证常用于比较同一算法不同参数配置以及不同算法的效果。
注意：这是人工核对出处，不表示你的检索器已经通过测试。

## 官方内容与本练习的边界
官方 C3：读取/清洗 PDF 和 Markdown，500 字切分与 50 字重叠，Embedding API、Chroma、相似度检索及 MMR；已有完整代码。
官方 C5：系统评估、检索错误分析与优化；不等同于现成的自动评分作业。
本说明额外安排：BM25、五题证据标注及 hit@3。官方 C3 并未实现 BM25/混合检索。

## 费用与版本
本包阅读、文本提取、BM25 检索不需要账户、付费 API、GPU。
原版 C3 使用 LangChain/Chroma，默认智谱 Embedding，也有其他服务示例；需要对应 API 权限，不能保证免费额度。完整问答还需要 LLM。
原课依赖锁在 LangChain 0.3.0 等旧版本，不能保证直接匹配最新 SDK。无需租阿里云服务器。
已检查 PDF 页数并用 pdftotext 提取原文、人工核对第一题出处；未安装/运行整门课程环境，未执行向量检索或 LLM 请求。
