# visualize.py
# 纯前端 vis-network 可视化
#   python visualize.py --kg ./HiddenKG/output/final_kg.json --out ./HiddenKG/output/final_kg.html
# 可选：
#   --toc_color "#3C7BE6" --entity_color "#2BB673" --toc_size 22 --entity_size 16

import argparse
import json
from pathlib import Path
from html import escape

def load_final_kg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_nodes_edges(kg: dict, toc_color: str, entity_color: str, toc_size: int, entity_size: int):
    nodes_out = []
    edges_out = []

    # 节点
    for n in kg.get("nodes", []):
        name = (n.get("name") or "").strip()
        ntype = (n.get("type") or "").strip().lower()
        desc = (n.get("description") or "").strip()
        if not name:
            continue
        is_toc = (ntype == "toc")
        nodes_out.append({
            "id": name,
            "label": name,
            "title": escape(desc) or name,
            "color": toc_color if is_toc else entity_color,
            "size": toc_size if is_toc else entity_size,
            "shape": "dot"
        })

    # 按关系类型设置更短的边长
    # 数值越小，视觉上越“紧凑”
    edge_length_map = {
        "toc→toc": 65,
        "toc→core": 60,
        "core→non-core": 60,
        "pred": 60,  # 预测边更短
        # 兜底
        "_default": 40,
    }

    # 边
    for e in kg.get("edges", []):
        src = (e.get("source") or "").strip()
        tgt = (e.get("target") or "").strip()
        rel = (e.get("relationship") or "").strip()
        rtype = (e.get("relation_type") or "").strip()
        if not src or not tgt or src == tgt:
            continue

        # 选择边长
        length = edge_length_map.get(rel, edge_length_map["_default"])

        # 基础边对象
        edge_obj = {
            "from": src,
            "to": tgt,
            "title": escape(f"{rel} | {rtype}") if (rel or rtype) else "",
            "length": length,
            "arrows": "to" if rel in ("core→non-core", "toc→core", "toc→toc") else ""
        }

        # pred 边：在边中间显示类型（relation_type）
        if rel == "pred" and rtype:
            edge_obj["label"] = rtype
            edge_obj["font"] = {"align": "middle", "size": 11}
            # 如果希望更醒目，可给 pred 边加不同宽度或虚线：
            # edge_obj["width"] = 1.5
            # edge_obj["dashes"] = True

        edges_out.append(edge_obj)

    return nodes_out, edges_out

def write_vis_html(nodes: list, edges: list, out_html: Path):
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Tree-KG Visualization</title>
  <style>
    html, body {{ height: 100%; margin: 0; }}
    #mynetwork {{ width: 100%; height: 100vh; border: 1px solid #ddd; }}
  </style>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
</head>
<body>
<div id="mynetwork"></div>
<script>
  const nodes = new vis.DataSet({json.dumps(nodes, ensure_ascii=False)});
  const edges = new vis.DataSet({json.dumps(edges, ensure_ascii=False)});

  const container = document.getElementById('mynetwork');
  const data = {{ nodes, edges }};
  const options = {{
    interaction: {{
      hover: true,
      tooltipDelay: 120,
      zoomView: true,
      dragView: true
    }},
    physics: {{
      enabled: true,
      stabilization: {{ iterations: 400, updateInterval: 25 }},
      barnesHut: {{
        gravitationalConstant: -30000,
        centralGravity: 0.18,
        springLength: 80,        // 全局弹簧长度更短
        springConstant: 0.05,
        damping: 0.10
      }}
    }},
    edges: {{
      smooth: true,
      color: {{ color: '#999', opacity: 0.85 }},
      width: 1,
      selectionWidth: 2,
      hoverWidth: 1.5,
      font: {{ size: 11, align: 'middle' }} // 让边标签默认居中
    }},
    nodes: {{
      font: {{ size: 13 }}
    }}
  }};

  const network = new vis.Network(container, data, options);

  network.once("stabilizationIterationsDone", function () {{
    network.fit({{ animation: true }});
  }});
</script>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"✅ 可视化生成：{out_html.resolve()}")

def main():
    ap = argparse.ArgumentParser(description="可视化 final_kg.json（vis-network）")
    ap.add_argument("--kg", type=str,
                    default="./output/final_kg.json",
                    help="final_kg.json 路径（默认：./output/final_kg.json）")
    ap.add_argument("--out", type=str,
                    default="./output/final_kg.html",
                    help="输出 HTML 路径（默认：./HiddenKG/output/final_kg.html）")
    ap.add_argument("--toc_color", type=str, default="#3C7BE6", help="TOC 节点颜色（默认蓝）")
    ap.add_argument("--entity_color", type=str, default="#2BB673", help="实体节点颜色（默认绿）")
    ap.add_argument("--toc_size", type=int, default=22, help="TOC 节点尺寸（默认 22）")
    ap.add_argument("--entity_size", type=int, default=16, help="实体节点尺寸（默认 16）")
    args = ap.parse_args()

    kg = load_final_kg(Path(args.kg))
    nodes, edges = build_nodes_edges(
        kg,
        toc_color=args.toc_color,
        entity_color=args.entity_color,
        toc_size=args.toc_size,
        entity_size=args.entity_size
    )
    write_vis_html(nodes, edges, Path(args.out))

if __name__ == "__main__":
    main()
