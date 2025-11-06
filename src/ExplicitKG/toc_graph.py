import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

def extract_toc_entities_and_relations(input_path: Path, output_path: Path):
    # 读取 TOC 数据
    with input_path.open("r", encoding="utf-8") as f:
        toc_data = json.load(f)

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    entity_count = 0
    relation_count = 0

    # 去重：按 section_id 记录已创建的节点（避免重复）
    created_by_section_id: Dict[str, str] = {}  # section_id -> node_name(title)

    # 递归处理每个顶层章节
    for section in toc_data:
        entity_count, relation_count = process_section(
            section=section,
            nodes=nodes,
            edges=edges,
            entity_count=entity_count,
            relation_count=relation_count,
            created_by_section_id=created_by_section_id,
            parent_node_name=None,  # 顶层无父节点
        )

    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "entity_count": entity_count,
        "relation_count": relation_count
    }

    # 写出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    print(f"知识图谱文件已保存至：{output_path.resolve()}")
    print(f"生成了 {entity_count} 个实体，{relation_count} 个关系。")


def process_section(
    section: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    entity_count: int,
    relation_count: int,
    created_by_section_id: Dict[str, str],
    parent_node_name: Optional[str] = None,
) -> Tuple[int, int]:
    """
    递归处理每一层的章节节点及其关系（仅 level 1/2/3）
    - 节点：用 title 作为 name
    - 边：用父节点的 name -> 子节点的 name
    """
    section_id = (section.get("id") or "").strip()
    section_title = (section.get("title") or "").strip()

    # level 可能是字符串，这里统一成 int
    try:
        section_level = int(section.get("level", 0) or 0)
    except Exception:
        section_level = 0

    # 只处理 level 1~3
    if section_level not in (1, 2, 3):
        return entity_count, relation_count

    # —— 创建/记录节点 ——（按 section_id 去重）
    if section_title:
        # 如果该 section_id 未创建过节点，则创建新节点
        if section_id not in created_by_section_id:
            created_by_section_id[section_id] = section_title  # 记录 name

            nodes.append({
                "name": section_title,               # 可视化或后续处理统一读 name
                "title": section_title,              # 冗余保留
                "level": f"level{section_level}",     # level1/level2/level3
                "description": section_id            # 用章节编号做描述
            })
            entity_count += 1

        # 当前节点名称（后续给子节点连边用）
        current_node_name = created_by_section_id[section_id]

        # —— 父子关系边 ——（父 name -> 子 name）
        if parent_node_name:
            edges.append({
                "source": parent_node_name,
                "target": current_node_name,
                "relationship": "child_of",
                "description": f"{parent_node_name} -> {current_node_name}"
            })
            relation_count += 1
    else:
        # 没标题就不创建节点，也不递归
        return entity_count, relation_count

    # —— 递归子节点 ——（同样只保留到 level 3）
    for child in section.get("children", []) or []:
        entity_count, relation_count = process_section(
            section=child,
            nodes=nodes,
            edges=edges,
            entity_count=entity_count,
            relation_count=relation_count,
            created_by_section_id=created_by_section_id,
            parent_node_name=current_node_name,  # 用 name 作为父节点标识
        )

    return entity_count, relation_count


def main():
    # 以脚本所在目录为基准：src/ExplicitKG
    script_dir = Path(__file__).resolve().parent
    in_file = script_dir / "output" / "toc_with_entities_and_relations.json"
    out_file = script_dir / "output" / "toc_graph.json"

    if not in_file.exists():
        raise FileNotFoundError(f"未找到输入文件：{in_file}")

    extract_toc_entities_and_relations(in_file, out_file)


if __name__ == "__main__":
    main()
