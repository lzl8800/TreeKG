from typing import Dict, List, Tuple
import json
import os
from ExplicitKG.config.config import OUTPUT_DIR as EXPLICIT_OUTPUT_DIR  # ExplicitKG的OUTPUT_DIR
from HiddenKG.config.config import OUTPUT_DIR as HIDDEN_OUTPUT_DIR  # HiddenKG的OUTPUT_DIR

# 读取实体文件并解析成对象
def read_entities(infile: str):
    with open(infile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("entities", {})


# 读取TOC层文件
def read_toc(infile: str):
    with open(infile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# 读取pred_result文件
def read_pred_result(infile: str):
    with open(infile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("edges", [])


# 提取核心节点和边
def extract_core_and_edges(entities: Dict[str, Dict], toc_data: Dict) -> Tuple[
    List[Dict], List[Dict], int, int, int, int]:
    core_nodes = []
    edges = []
    toc_nodes = toc_data.get("nodes", [])
    toc_edges = toc_data.get("edges", [])

    toc_node_count = len(toc_nodes)
    toc_edge_count = len(toc_edges)

    core_count = 0
    non_core_count = 0

    # 创建名称到节点 ID 的映射
    toc_name_to_id = {node['name']: node['id'] for node in toc_nodes}

    # 为所有节点分配唯一的 ID
    node_id_counter = 1

    # 计数没有找到连接的核心节点数量
    unconnected_core_count = 0

    # 提取核心节点
    for entity_name, entity in entities.items():
        if 'name' not in entity:
            print(f"Warning: Entity {entity_name} is missing 'name' key, skipping.")
            continue

        node = {
            "id": node_id_counter,  # 为核心节点分配新的唯一 ID
            "name": entity["name"],
            "type": entity.get("type", ""),
            "description": entity.get("updated_description", entity.get("original", ""))
        }
        node_id_counter += 1  # 每分配一个 ID，递增

        if entity.get("role") == "core":
            core_nodes.append(node)
            core_count += 1
        else:
            non_core_count += 1

        # 查找每个core节点对应的TOC小节
        core_node = node
        connected_to_toc = False  # 标记该核心节点是否连接到TOC

        for occurrence in entity.get("occurrences", []):
            occurrence_title = occurrence.get("title", "")
            if occurrence_title:
                # 根据occurrences中的title找到对应的TOC节点
                for toc_node in toc_nodes:
                    if toc_node["title"] == occurrence_title:  # 完全匹配
                        edge = {
                            "source": toc_node["id"],
                            "target": core_node["id"],
                            "relationship": "core-toc",  # core节点与TOC节点之间的关系
                            "relation_type": "所属小节"
                        }
                        edges.append(edge)
                        connected_to_toc = True
                        break  # 找到匹配的小节就跳出循环

        # 如果核心节点没有找到匹配的TOC节点，更新未连接核心节点计数
        if not connected_to_toc:
            unconnected_core_count += 1
            print(f"核心节点 '{core_node['name']}' 没有找到TOC连接")  # 只打印没有找到连接的核心节点

        # 处理实体的邻居，提取边
        for neighbor in entity.get("neighbors", []):
            target = neighbor.get("name")
            snippet = neighbor.get("snippet", "")
            if not target:
                continue  # 如果邻居没有名称，跳过

            # 区分核心节点和非核心节点之间的边
            if entity.get("role") == "core" and target in entities:
                edge = {
                    "source": node["id"],  # 使用新的 ID
                    "target": toc_name_to_id.get(target, target),  # 查找 TOC 中父节点的 ID
                    "relationship": "core-core" if entities[target].get("role") == "core" else "core-non-core",
                    "relation_type": snippet.split("|")[0] if "|" in snippet else snippet
                }
                edges.append(edge)

    # 为每个核心节点找到其上一层TOC节点并建立边
    for core_node in core_nodes:
        connected_to_toc = False
        for toc_node in toc_nodes:
            # 如果TOC节点的name与核心节点的name匹配，则认为这是父节点
            if core_node["name"] == toc_node["name"]:
                # 为核心节点和上一层TOC节点之间建立边
                edge = {
                    "source": toc_node["id"],
                    "target": core_node["id"],
                    "relationship": "toc-core",
                    "relation_type": "上一级章节"
                }
                edges.append(edge)
                connected_to_toc = True

        if not connected_to_toc:
            unconnected_core_count += 1

    # 为TOC节点分配唯一ID
    for toc_node in toc_nodes:
        toc_node["id"] = node_id_counter
        node_id_counter += 1

    return toc_nodes + core_nodes, edges, toc_node_count, toc_edge_count, core_count, non_core_count


# 将数据转换为标准的知识图谱格式并保存
def convert_to_kg_format(entities: Dict[str, Dict], toc_data: Dict, pred_edges: List[Dict], out_path: str):
    # 提取核心节点和边
    nodes, edges, toc_node_count, toc_edge_count, core_count, non_core_count = extract_core_and_edges(
        entities, toc_data)

    # 将TOC的节点和边完全合并到最终的节点和边中
    toc_nodes = toc_data.get("nodes", [])
    toc_edges = toc_data.get("edges", [])

    # 确保TOC节点和边被完整合并
    nodes.extend(toc_nodes)
    edges.extend(toc_edges)

    # 合并 pred_result 中的边数据
    edges.extend(pred_edges)

    # 构建知识图谱格式的 JSON
    kg = {
        "nodes": nodes,
        "edges": edges
    }

    # 保存到输出文件
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

    # 打印统计信息并加入到最终生成的文件中
    print(f"知识图谱文件已保存至：{os.path.abspath(out_path)}")
    print(f"TOC 层节点数量：{toc_node_count}")
    print(f"TOC 层边数量：{toc_edge_count}")
    print(f"核心节点数量：{core_count}")
    print(f"非核心节点数量：{non_core_count}")
    print(f"总边数量：{len(edges)}")


# 主流程
def main():
    # 输入输出文件路径（通过 config.py 配置）
    entities_in = HIDDEN_OUTPUT_DIR / "dedup_result.json"  # 这里填入你的结果文件路径
    toc_in = EXPLICIT_OUTPUT_DIR / "toc_graph.json"  # TOC 文件路径
    pred_in = HIDDEN_OUTPUT_DIR / "pred_result.json"  # Pred 结果文件路径
    output_path = HIDDEN_OUTPUT_DIR / "final_kg.json"  # 输出的知识图谱文件路径

    # 读取结果文件
    entities = read_entities(entities_in)

    # 读取TOC数据
    toc_data = read_toc(toc_in)

    # 读取Pred结果文件
    pred_edges = read_pred_result(pred_in)

    # 转换为知识图谱格式并保存
    convert_to_kg_format(entities, toc_data, pred_edges, output_path)


if __name__ == "__main__":
    main()
