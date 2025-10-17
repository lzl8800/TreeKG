# README

## 项目名称：TreeKG

### 项目概述：
TreeKG 项目旨在构建一个基于教科书的知识图谱，使用显式和隐式方法生成结构化的章节和实体关系图谱。该项目遵循 **清华大学 TreeKG 论文** 提供的思路，依赖自然语言处理和知识图谱构建技术，通过分阶段的处理步骤生成最终的 **final_kg.json**。

### 主要步骤：
本项目分为两个核心阶段，具体步骤如下：

---

## 核心阶段一：**初始构建（显式 KG）**

### 目标：
构建一个基于教科书天然层级的知识图谱，生成 **"章节 - 实体"** 的结构化骨架。

### 步骤 1：**文本分割（Text Segmentation）**
- **核心逻辑**：通过正则表达式从 PDF 目录中提取章节层级关系。根据正则表达式，匹配“章 - 节 - 小节”边界，生成层级化的 **TOC 节点（TOC Node）**。
  - 示例正则：
    - `(第\\d+章.*\\n\\n)`：匹配 “第 1 章 电场 \n\n” 作为 2 级节点；
    - `(\\*?\\d+\\.\\d+.*\\n\\n)`：匹配 “1.1 电荷 \n\n” 作为 3 级节点。
  - **输出**：层级关系的 TOC 节点列表，每个节点带有 `id`（如 “section_1.1”）、`title`（章节名）、`level`（层级）、`page_start`/`page_end`（PDF 页码）等信息。

### 步骤 2：**自底向上摘要（Bottom-Up Summarization）**
- **核心逻辑**：从 **最小层级节点**（如 3 级小节）开始生成摘要，并逐步向上聚合生成上层节点的摘要。
  - **LLM Prompt**：为每个小节生成200-300字的摘要，确保包含核心概念、关键定理和领域术语。
  - **输出**：每个 TOC 节点与一份摘要关联，用于后续的实体提取。

### 步骤 3：**实体与关系提取（Entity & Relation Extraction）**
- **核心逻辑**：从小节摘要中提取领域实体（名称、别名、类型和原始描述），并基于摘要与实体提取实体间关系。
  - **实体提取**：提取实体并生成对应的 JSON 文件。
  - **关系提取**：识别实体之间的关系（如定义、依赖、应用等），并为每个小节建立边。
  - **输出**：实体列表和关系列表，生成显式知识图谱结构。

### 步骤 4：**树状结构组装（Tree-like Graph Construction）**
- **核心逻辑**：将步骤 1 到步骤 3 中的 TOC 节点、实体节点和边整合，生成“树状层次图”（显式 KG）。
  - 输出知识图谱的标准 JSON 格式，示例如下：
    ```json
    {
      "nodes": [
        {"id": "section_1.1", "type": "toc", "level": 3, "title": "电荷", "summary": "..."},
        {"id": "entity_1", "type": "entity", "name": "电荷", "alias": [], "type": "物理概念", "description": "...", "section_id": "section_1.1", ...}
      ],
      "edges": [
        {"source": "section_1", "target": "section_1.1", "type": "has_subsection"},
        {"source": "section_1.1", "target": "entity_1", "type": "has_entity"}
      ]
    }
    ```

---

## 核心阶段二：**迭代扩展（隐式 KG）**

### 目标：
基于显式 KG 扩展隐式知识图谱，通过预定义的操作符挖掘跨章节关系。

### 操作符 1：**上下文卷积（Contextual-based Convolution）**
- **核心目标**：增强实体描述，通过邻居上下文信息补全实体描述，提升语义完整性。
- **LLM Prompt**：根据实体的邻居实体和关系，增强实体描述。

### 操作符 2：**实体聚合（Entity Aggregation）**
- **核心目标**：将实体分配 **core** 和 **non-core** 角色，简化层级结构。
- **LLM Prompt**：判断每个实体的核心程度（核心实体 vs 辅助实体）。

### 操作符 3：**节点嵌入（Node Embedding）**
- **核心目标**：将实体描述转换为稠密向量，进行相似性检索和边预测。
- **使用工具**：Sentence-BERT 模型（如 all-MiniLM-L6-v2）。

### 操作符 4：**实体去重（Entity Deduplication）**
- **核心目标**：识别名称不同但语义相同的实体，消除冗余。
- **实现步骤**：基于实体嵌入向量，通过 LLM 进一步确认实体是否相同。

### 操作符 5：**边预测（Edge Prediction）**
- **核心目标**：预测实体间的潜在关系，补充水平边（`entity_related`）。
- **实现步骤**：综合语义相似性、结构关联性等因素预测潜在关系。

---

## 运行步骤

### 准备数据与模型

- 将**教材源文件**（PDF/Docx 等）放到：`src/ExplicitKG/output/`
  
- 将 **BERT 模型文件夹** 放到：`src/HiddenKG/model/`  
  （例如：`src/HiddenKG/model/bert-base-chinese/`，内部包含 `config.json`, `pytorch_model.bin`, `vocab.txt` 等）

### 文件顺序：

1. **运行 `ExplicitKG` 阶段**：
   - 执行 `ExplicitKG` 下的 `main.py`，该文件负责数据预处理、实体提取和关系提取。

2. **运行 `HiddenKG` 阶段**：
   - 执行 `HiddenKG` 下的 `main.py`，该文件负责隐式知识图谱构建、实体去重、边预测等操作，生成最终的知识图谱。


### 输出：
最终生成的知识图谱文件为 **`final_kg.json`**，格式为标准的 **G=(V, E)** 知识图谱，其中：
- **V**：包含 TOC 节点和实体节点。
- **E**：包含 TOC 节点之间、TOC 节点与实体节点之间的边。

---


## 如何运行：

1. 确保已经安装了所有必要的依赖：
   


2. 在命令行中运行以下命令：
   - **运行 `ExplicitKG` 阶段**：
     ```bash
     python src/ExplicitKG/main.py
     ```

   - **运行 `HiddenKG` 阶段**：
     ```bash
     python src/HiddenKG/main.py
     ```

### 结果：
- 输出文件 **`final_kg.json`** 将保存在 `src/HiddenKG/output` 文件夹中。

---

## 联系方式：
- 代码版权：五邑大学电子与信息工程 李子亮
- 联系邮箱：lzl8800@foxmail.com

