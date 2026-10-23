"""
Obsidian Knowledge Graph Visualizer Module.
Parses markdown files and [[wikilinks]] across Vault/ to build an interactive,
cyber-themed vis.js network graph for Streamlit.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import config


def get_vault_graph_data() -> Dict[str, Any]:
    """
    Scans Vault/ directory, parses all markdown files, frontmatter, and [[wikilinks]].
    Returns a dict with 'nodes', 'edges', and 'stats'.
    """
    vault_dir = Path(config.VAULT_DIR).resolve()
    if not vault_dir.exists():
        return {"nodes": [], "edges": [], "stats": {}}

    md_files = list(vault_dir.rglob("*.md"))
    
    # Node id -> data
    nodes_map: Dict[str, Dict[str, Any]] = {}
    edges_set = set()
    raw_edges = []

    # Helper to derive clean node ID and title
    for f in md_files:
        rel_path = f.relative_to(vault_dir)
        folder = rel_path.parts[0] if len(rel_path.parts) > 1 else "Root"
        stem = f.stem

        # Default classification
        group = "theory"
        color_border = "#818cf8"
        color_bg = "rgba(99, 102, 241, 0.85)"
        
        if "Paper_Cards" in folder or re.match(r"^\d{4}\.\d{4,5}$", stem):
            group = "paper"
            color_border = "#38bdf8"
            color_bg = "rgba(14, 165, 233, 0.85)"
        elif "Profile" in folder:
            group = "profile"
            color_border = "#f59e0b"
            color_bg = "rgba(245, 158, 11, 0.85)"
        elif "Drafts" in folder:
            group = "draft"
            color_border = "#94a3b8"
            color_bg = "rgba(148, 163, 184, 0.6)"

        # Read content to extract title and preview
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content = ""

        # Extract title
        title_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        if title_match:
            display_title = title_match.group(1).strip()
        else:
            display_title = stem.replace("_", " ")

        # Short preview for tooltip
        clean_lines = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("#") and not l.startswith("---") and not l.startswith("-")]
        preview = " ".join(clean_lines[:3])[:220] if clean_lines else "Akademik kavram / makale kütüğü"

        nodes_map[stem] = {
            "id": stem,
            "label": display_title[:28] + ("..." if len(display_title) > 28 else ""),
            "full_title": display_title,
            "group": group,
            "folder": folder,
            "preview": preview,
            "color": {
                "background": color_bg,
                "border": color_border,
                "highlight": {
                    "background": "#ec4899",
                    "border": "#f43f5e"
                }
            },
            "font": {
                "color": "#f8fafc",
                "size": 13,
                "face": "Inter, sans-serif"
            },
            "degree": 0,
            "file_path": str(f)
        }

    # Second pass: Extract [[wikilinks]] to create edges
    for f in md_files:
        stem = f.stem
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content = ""

        wikilinks = re.findall(r"\[\[(.*?)\]\]", content)
        for link in wikilinks:
            # Handle aliases like [[Target|Display Name]]
            target = link.split("|")[0].strip()
            # Clean possible markdown or subpaths
            target = Path(target).stem

            if not target:
                continue

            # If target node doesn't exist as a file, register it as a virtual concept node
            if target not in nodes_map:
                nodes_map[target] = {
                    "id": target,
                    "label": target.replace("_", " ")[:24],
                    "full_title": target.replace("_", " "),
                    "group": "concept",
                    "folder": "Concepts",
                    "preview": f"Bağlantılı kavram düğümü: {target}",
                    "color": {
                        "background": "rgba(168, 85, 247, 0.7)",
                        "border": "#a855f7",
                        "highlight": {"background": "#ec4899", "border": "#f43f5e"}
                    },
                    "font": {"color": "#e2e8f0", "size": 11, "face": "Inter, sans-serif"},
                    "degree": 0,
                    "file_path": ""
                }

            # Avoid self-loops and duplicate edges
            if stem != target:
                edge_key = tuple(sorted([stem, target]))
                if edge_key not in edges_set:
                    edges_set.add(edge_key)
                    raw_edges.append({
                        "from": stem,
                        "to": target,
                        "color": {"color": "rgba(148, 163, 184, 0.35)", "highlight": "#38bdf8"},
                        "arrows": "to",
                        "smooth": {"type": "continuous", "roundness": 0.2}
                    })
                    nodes_map[stem]["degree"] += 1
                    nodes_map[target]["degree"] += 1

    # Adjust node size dynamically based on degree (hub notes appear larger)
    nodes_list = []
    for n in nodes_map.values():
        base_size = 14
        size = base_size + min(n["degree"] * 3.5, 30)
        n["size"] = size
        nodes_list.append(n)

    # Top connected hub concepts
    sorted_hubs = sorted(nodes_list, key=lambda x: x["degree"], reverse=True)[:5]
    hub_info = [{"title": h["full_title"], "degree": h["degree"]} for h in sorted_hubs]

    stats = {
        "total_nodes": len(nodes_list),
        "total_edges": len(raw_edges),
        "papers_count": len([n for n in nodes_list if n["group"] == "paper"]),
        "theory_count": len([n for n in nodes_list if n["group"] == "theory"]),
        "top_hubs": hub_info
    }

    return {
        "nodes": nodes_list,
        "edges": raw_edges,
        "stats": stats
    }


def generate_graph_html(graph_data: Dict[str, Any], height: int = 700) -> str:
    """
    Renders an interactive vis-network HTML widget with physics simulation,
    search filter, and a click-to-preview sidebar modal.
    """
    nodes_json = json.dumps(graph_data.get("nodes", []))
    edges_json = json.dumps(graph_data.get("edges", []))

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            body, html {{
                width: 100%;
                height: 100%;
                background-color: #0b0f19;
                color: #e2e8f0;
                overflow: hidden;
            }}
            #graph-container {{
                width: 100%;
                height: 100%;
                position: relative;
            }}
            #network {{
                width: 100%;
                height: 100%;
                background: radial-gradient(circle at 50% 50%, #111827 0%, #080c14 100%);
            }}
            /* Floating Control Toolbar */
            .graph-toolbar {{
                position: absolute;
                top: 16px;
                left: 16px;
                display: flex;
                gap: 10px;
                z-index: 100;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 8px 14px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            }}
            .search-box {{
                background: rgba(30, 41, 59, 0.7);
                border: 1px solid rgba(56, 189, 248, 0.3);
                border-radius: 8px;
                padding: 6px 12px;
                color: #f8fafc;
                font-size: 13px;
                outline: none;
                width: 200px;
                transition: all 0.2s;
            }}
            .search-box:focus {{
                border-color: #38bdf8;
                width: 250px;
                box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
            }}
            .filter-btn {{
                background: rgba(30, 41, 59, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: #94a3b8;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 12px;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .filter-btn:hover, .filter-btn.active {{
                background: #38bdf8;
                color: #0b0f19;
                font-weight: 600;
            }}
            /* Legend Bar */
            .graph-legend {{
                position: absolute;
                bottom: 16px;
                left: 16px;
                display: flex;
                gap: 16px;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                padding: 8px 16px;
                border-radius: 30px;
                font-size: 12px;
                color: #cbd5e1;
                z-index: 100;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 6px;
            }}
            .dot {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
            }}
            /* Details Preview Drawer */
            #node-drawer {{
                position: absolute;
                top: 16px;
                right: 16px;
                width: 320px;
                max-height: 85%;
                background: rgba(15, 23, 42, 0.95);
                backdrop-filter: blur(16px);
                border: 1px solid rgba(56, 189, 248, 0.3);
                border-radius: 14px;
                padding: 18px;
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
                z-index: 101;
                display: none;
                flex-direction: column;
                gap: 12px;
                overflow-y: auto;
            }}
            #drawer-title {{
                font-size: 16px;
                font-weight: 700;
                color: #38bdf8;
                line-height: 1.3;
            }}
            #drawer-badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                width: fit-content;
            }}
            #drawer-preview {{
                font-size: 13px;
                color: #94a3b8;
                line-height: 1.5;
                background: rgba(30, 41, 59, 0.5);
                padding: 10px;
                border-radius: 8px;
            }}
            #drawer-close {{
                position: absolute;
                top: 12px;
                right: 12px;
                background: transparent;
                border: none;
                color: #94a3b8;
                font-size: 16px;
                cursor: pointer;
            }}
            #drawer-close:hover {{
                color: #fff;
            }}
            .ask-btn {{
                background: linear-gradient(135deg, #38bdf8 0%, #6366f1 100%);
                border: none;
                border-radius: 8px;
                color: #fff;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: opacity 0.2s;
            }}
            .ask-btn:hover {{
                opacity: 0.9;
            }}
        </style>
    </head>
    <body>
        <div id="graph-container">
            <div class="graph-toolbar">
                <input type="text" id="search-input" class="search-box" placeholder="Kavram / Makale Ara...">
                <button class="filter-btn active" onclick="filterGroup('all')">Tümü</button>
                <button class="filter-btn" onclick="filterGroup('paper')">Makaleler</button>
                <button class="filter-btn" onclick="filterGroup('theory')">Teori</button>
                <button class="filter-btn" onclick="resetPhysics()">Düzenle</button>
            </div>

            <div id="network"></div>

            <div class="graph-legend">
                <div class="legend-item"><span class="dot" style="background:#38bdf8;"></span> Makale Kartları</div>
                <div class="legend-item"><span class="dot" style="background:#818cf8;"></span> Temel Teori</div>
                <div class="legend-item"><span class="dot" style="background:#a855f7;"></span> Sinaptik Kavramlar</div>
                <div class="legend-item"><span class="dot" style="background:#f59e0b;"></span> Profil</div>
            </div>

            <!-- Detail Drawer -->
            <div id="node-drawer">
                <button id="drawer-close" onclick="closeDrawer()">✕</button>
                <span id="drawer-badge"></span>
                <h3 id="drawer-title">Kavram Başlığı</h3>
                <div id="drawer-preview">Özet yükleniyor...</div>
                <div style="font-size: 12px; color: #64748b;" id="drawer-degree">Bağlantı Sayısı: 0</div>
                <button class="ask-btn" id="drawer-action-btn" onclick="sendToParentChat()">💬 Bu Kavramı Sohbete Sor</button>
            </div>
        </div>

        <script>
            const rawNodes = {nodes_json};
            const rawEdges = {edges_json};

            const nodesDataSet = new vis.DataSet(rawNodes);
            const edgesDataSet = new vis.DataSet(rawEdges);

            const container = document.getElementById('network');
            const data = {{
                nodes: nodesDataSet,
                edges: edgesDataSet
            }};

            const options = {{
                nodes: {{
                    shape: 'dot',
                    scaling: {{
                        min: 14,
                        max: 38
                    }},
                    borderWidth: 2,
                    shadow: {{
                        enabled: true,
                        color: 'rgba(0,0,0,0.5)',
                        size: 8,
                        x: 2,
                        y: 2
                    }}
                }},
                edges: {{
                    width: 1.2,
                    color: {{
                        inherit: false,
                        opacity: 0.35
                    }},
                    selectionWidth: 2.5
                }},
                physics: {{
                    solver: 'forceAtlas2Based',
                    forceAtlas2Based: {{
                        gravitationalConstant: -45,
                        centralGravity: 0.015,
                        springLength: 90,
                        springConstant: 0.08,
                        damping: 0.4
                    }},
                    stabilization: {{
                        iterations: 150
                    }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 100,
                    zoomView: true,
                    dragView: true
                }}
            }};

            const network = new vis.Network(container, data, options);
            let selectedNodeData = null;

            // Click node to open drawer
            network.on('click', function(params) {{
                if (params.nodes.length > 0) {{
                    const nodeId = params.nodes[0];
                    const node = rawNodes.find(n => n.id === nodeId);
                    if (node) {{
                        selectedNodeData = node;
                        showDrawer(node);
                    }}
                }} else {{
                    closeDrawer();
                }}
            }});

            function showDrawer(node) {{
                const drawer = document.getElementById('node-drawer');
                const title = document.getElementById('drawer-title');
                const badge = document.getElementById('drawer-badge');
                const preview = document.getElementById('drawer-preview');
                const degree = document.getElementById('drawer-degree');

                title.innerText = node.full_title;
                preview.innerText = node.preview;
                degree.innerText = '⚡ Sinaps (Bağlantı Sayısı): ' + node.degree;

                if (node.group === 'paper') {{
                    badge.innerText = 'Akademik Makale';
                    badge.style.background = 'rgba(56, 189, 248, 0.2)';
                    badge.style.color = '#38bdf8';
                }} else if (node.group === 'theory') {{
                    badge.innerText = 'Temel Teori Notu';
                    badge.style.background = 'rgba(129, 140, 248, 0.2)';
                    badge.style.color = '#818cf8';
                }} else {{
                    badge.innerText = 'Sinaptik Kavram';
                    badge.style.background = 'rgba(168, 85, 247, 0.2)';
                    badge.style.color = '#a855f7';
                }}

                drawer.style.display = 'flex';
            }}

            function closeDrawer() {{
                document.getElementById('node-drawer').style.display = 'none';
            }}

            function sendToParentChat() {{
                if (!selectedNodeData) return;
                const prompt = selectedNodeData.group === 'paper' 
                    ? `[${{selectedNodeData.id}}] ${{selectedNodeData.full_title}} makalesini, çözdüğü problemi ve ana katkılarını açıkla.`
                    : `${{selectedNodeData.full_title}} kavramının teorik motivasyonunu, formüllerini ve PyTorch implementasyonunu açıkla.`;

                // Post message to Streamlit chat input
                const parentDoc = window.parent.document;
                const chatInput = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
                if (chatInput) {{
                    const nativeSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, "value").set;
                    nativeSetter.call(chatInput, prompt);
                    chatInput.dispatchEvent(new Event("input", {{ bubbles: true }}));
                    chatInput.focus();
                }}
                closeDrawer();
            }}

            // Search filter
            document.getElementById('search-input').addEventListener('input', function(e) {{
                const val = e.target.value.toLowerCase().trim();
                if (!val) {{
                    network.fit();
                    return;
                }}
                const matched = rawNodes.find(n => n.full_title.toLowerCase().includes(val) || n.id.toLowerCase().includes(val));
                if (matched) {{
                    network.focus(matched.id, {{
                        scale: 1.4,
                        animation: {{
                            duration: 500,
                            easingFunction: 'easeInOutQuad'
                        }}
                    }});
                    network.selectNodes([matched.id]);
                    showDrawer(matched);
                }}
            }});

            function filterGroup(group) {{
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                event.target.classList.add('active');

                if (group === 'all') {{
                    nodesDataSet.update(rawNodes.map(n => ({{ id: n.id, hidden: false }})));
                }} else {{
                    nodesDataSet.update(rawNodes.map(n => ({{
                        id: n.id,
                        hidden: n.group !== group
                    }})));
                }}
            }}

            function resetPhysics() {{
                network.stabilize();
                network.fit();
            }}
        </script>
    </body>
    </html>
    """
    return html
