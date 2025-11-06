# 纯前端 vis-network 可视化
import argparse
import json
from pathlib import Path
from html import escape

def load_final_kg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_nodes_edges(kg: dict, toc_color: str, entity_color: str, toc_size: int, entity_size: int):
    # -------- 节点（去重合并）--------
    nodes_out = []
    node_idx = {}  # name -> index in nodes_out

    for n in kg.get("nodes", []):
        name = (n.get("name") or "").strip()
        ntype = (n.get("type") or "").strip().lower()
        desc  = (n.get("description") or "").strip()
        if not name:
            continue

        is_toc = (ntype == "toc")
        color  = toc_color if is_toc else entity_color
        size   = toc_size if is_toc else entity_size
        title  = escape(desc) if desc else name

        if name in node_idx:
            i = node_idx[name]
            old = nodes_out[i]
            was_toc = (old.get("color") == toc_color)
            now_toc = is_toc
            if (not was_toc) and now_toc:
                old["color"] = toc_color
                old["size"]  = toc_size
            old_title = old.get("title") or ""
            if len(title) > len(old_title):
                old["title"] = title
            nodes_out[i] = old
        else:
            node_idx[name] = len(nodes_out)
            nodes_out.append({
                "id": name,
                "label": name,
                "title": title,
                "color": color,
                "size": size,
                "shape": "dot"
            })

    # -------- 边（默认不显示 label，仅点击时显示）--------
    edges_out = []
    edge_label_map = {}  # id -> label to show on click

    edge_length_map = {
        "toc->toc": 65,
        "toc->core": 60,
        "core->non-core": 60,
        "pred": 60,
        "_default": 40,
    }

    for idx, e in enumerate(kg.get("edges", [])):
        src  = (e.get("source") or "").strip()
        tgt  = (e.get("target") or "").strip()
        et   = (e.get("type") or "").strip()
        desc = (e.get("description") or "").strip()
        if not src or not tgt or src == tgt:
            continue

        length = edge_length_map.get(et, edge_length_map["_default"])
        label_txt = f"{et} | {desc}" if desc else et

        # 稳定 id（避免中文分隔符歧义，使用管道和序号）
        eid = f"{src}|{tgt}|{et}|{idx}"

        # 初始不显示 label，仅保留 title（hover 提示仍可见）
        edge_obj = {
            "id": eid,
            "from": src,
            "to": tgt,
            "title": escape(label_txt) if label_txt else "",
            "length": length,
            "arrows": "to" if et in ("core->non-core", "toc->core", "toc->toc") else ""
        }
        edges_out.append(edge_obj)

        # 记录点击后要显示的文本
        edge_label_map[eid] = label_txt

    return nodes_out, edges_out, edge_label_map

def write_vis_html(nodes: list, edges: list, edge_label_map: dict, out_html: Path):
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
  const edgeLabelMap = {json.dumps(edge_label_map, ensure_ascii=False)};

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
        springLength: 80,
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
      font: {{ size: 11, align: 'middle' }}
    }},
    nodes: {{
      font: {{ size: 13 }}
    }}
  }};

  const network = new vis.Network(container, data, options);

  // 选中边时显示label；取消选择时清除label
  network.on('selectEdge', function (params) {{
    const ids = params.edges || [];
    const updates = ids.map(id => ({{ id, label: edgeLabelMap[id] || '' }}));
    if (updates.length) edges.update(updates);
  }});

  network.on('deselectEdge', function (params) {{
    const ids = (params.previousSelection && params.previousSelection.edges) ? params.previousSelection.edges : [];
    const updates = ids.map(id => ({{ id, label: '' }}));
    if (updates.length) edges.update(updates);
  }});

  // 当选中节点导致之前选中边丢失高亮时，也清掉边label
  network.on('selectNode', function (params) {{
    const sel = network.getSelection();
    const edgeIds = edges.getIds();
    // 清除所有边的label（成本很低）
    const updates = edgeIds.map(id => ({{ id, label: '' }}));
    if (updates.length) edges.update(updates);
  }});

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
    nodes, edges, edge_label_map = build_nodes_edges(
        kg,
        toc_color=args.toc_color,
        entity_color=args.entity_color,
        toc_size=args.toc_size,
        entity_size=args.entity_size
    )
    write_vis_html(nodes, edges, edge_label_map, Path(args.out))

if __name__ == "__main__":
    main()
