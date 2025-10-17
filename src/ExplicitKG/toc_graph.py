import json
from pathlib import Path

# 提取TOC层实体和关系
def extract_toc_entities_and_relations(input_file: str, output_file: str):
    # 读取TOC数据
    with open(input_file, "r", encoding="utf-8") as f:
        toc_data = json.load(f)

    # 存储图谱的节点和边
    nodes = []
    edges = []

    entity_count = 0  # 记录实体数量
    relation_count = 0  # 记录关系数量
    node_id_map = {}  # 用于为每个节点生成唯一ID

    # 遍历TOC层级数据，处理每一层
    for section in toc_data:
        # 递归处理章节中的子节点
        entity_count, relation_count = process_section(section, nodes, edges, entity_count, relation_count, node_id_map)

    # 创建标准知识图谱格式的输出
    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "entity_count": entity_count,
        "relation_count": relation_count
    }

    # 输出到指定文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    print(f"知识图谱文件已保存至：{output_path.resolve()}")
    print(f"生成了 {entity_count} 个实体，{relation_count} 个关系。")


def process_section(section, nodes, edges, entity_count, relation_count, node_id_map, parent_id=None):
    """递归处理每一层的章节节点及其关系"""
    section_id = section.get("id", "")
    section_title = section.get("title", "")
    section_level = section.get("level", "")

    # 只处理level 1 到 level 3 的节点
    if section_level not in [1, 2, 3]:
        return entity_count, relation_count

    # 处理实体（节点）
    if section_title:
        # 为每个节点生成一个唯一ID
        node_id = len(node_id_map) + 1
        node_id_map[section_id] = node_id

        nodes.append({
            "id": node_id,
            "name": section_title,  # 节点的name是title
            "title": section_title,
            "type": f"level{section_level}",  # 节点的type是level1, level2, level3
            "description": section_id  # 节点的描述是id
        })
        entity_count += 1

    # 处理关系（父子关系）
    if parent_id:
        edges.append({
            "source": parent_id,
            "target": node_id,
            "relationship": "child_of",  # 表示层级关系
            "description": f"{parent_id} -> {node_id}"  # 关系描述
        })
        relation_count += 1

    # 递归处理子节点（但不处理超过level 3的子节点）
    for child in section.get("children", []):
        entity_count, relation_count = process_section(child, nodes, edges, entity_count, relation_count, node_id_map, node_id)

    return entity_count, relation_count


# 主流程
def main():
    input_file = "output/toc_with_entities_and_relations.json"  # 输入文件路径
    output_file = "output/toc_graph.json"  # 输出文件路径
    extract_toc_entities_and_relations(input_file, output_file)


if __name__ == "__main__":
    main()
