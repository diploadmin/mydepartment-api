"""
LangGraph node for Neo4j graph expansion.

This is Option C from the architecture document — a standalone node
inserted between direct_retrieval and generate_final:

    START → direct_retrieval_node → ★ graph_expansion_node ★ → generate_final_node → END

The node reads retrieved documents from the state, enriches them with
graph context, and returns a replacement ToolMessage (same id) so the
add_messages reducer replaces instead of appending.

Graph context format is controlled by GRAPH_CONTEXT_FORMAT:
    "none"         — metadata enrichment only, no prompt injection
    "appended"     — single block appended after all sources (legacy)
    "inline"       — compact [Graph: ...] line under each source
    "boost_inline" — graph-boosted retrieval re-ranking + inline prompt injection
    "hint"         — conditional short hints only when text lacks graph info
"""

import logging
import os
from typing import Any

from langchain_core.messages import ToolMessage

from app.core.neo4j_config import (
    GRAPH_EXPANSION_ENABLED,
    GRAPH_CONTEXT_FORMAT,
    GRAPH_BOOST_ALPHA,
    GRAPH_BOOST_POOL_K,
)
from app.core.retrieval_context import get_param
from app.ai.ai_services.graph_expansion import GraphExpansionService

logger = logging.getLogger(__name__)

SOURCE_SEPARATOR = "\n\n---\n\n"

last_graph_data: dict | None = None

# context: this is the node that enriches the retrieval ToolMessage with graph context - called after direct_retrieval_node is finished 
# description: function to enrich the retrieval ToolMessage with graph context
# input given by: class State from graph.py - state is the output from direct_retrieval_node - saved in state['messages']
# parameters: state - the state of the LangGraph - this is the output from direct_retrieval_node
# return: dict[str, Any] - this is the output from the graph_expansion_node - this is the output for the LangGraph state reducer
async def graph_expansion_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node that enriches the retrieval ToolMessage with graph context.

    Prerequisites in state:
        - state["messages"]: must contain a ToolMessage with retrieved chunks
        - state["_retrieved_docs"]: list[Document] from retriever (set by direct_retrieval_node)
        - state["_site"]: site name for Neo4j database selection
        - state["_question"]: user question (for boost_inline and hint)

    Returns a delta dict for the LangGraph state reducer:
        - {'messages': [replacement_ToolMessage]} on success
        - {} (empty delta) when skipping — passes state through unchanged
    """
    # check if the graph expansion is enabled - this is set in config.py - default is True
    enabled = get_param('enable_graph_expansion', GRAPH_EXPANSION_ENABLED)
    if not enabled:
        return {}

    # import singleton.py - this is the class that handles the graph expansion service - this is the instance of the graph expansion service
    from app.core.singleton import get_graph_expansion_service
    # get the instance of the graph class GraphExpansionService and put it in the service variable
    service: GraphExpansionService = get_graph_expansion_service()
    if service is None:
        return {}

    # get the retrieved documents from the state - this is the return value from async direct_retrieval_node - saved in state['_retrieved_docs']
    docs = state.get("_retrieved_docs", [])
    # check if there are any documents - if not, return empty dict
    if not docs:
        logger.debug("graph_expansion_node: no docs in state, skipping")
        return {}

    # get the site name from the state - this is the output from direct_retrieval_node - saved in state['_site']
    site = state.get("_site", "diplomacy.edu")
    # get the user question from the state - this is the return value from async direct_retrieval_node - saved in state['_question']
    question = state.get("_question", "")
    # get the max relations from the config - this is neo4j config - default is 30
    max_relations = get_param('graph_max_relations', None)
    # put output of async function expand_documents into expansions and elapsed
    # expansions is parent document hash and hashed url variants
    # elapsed is the time taken to expand the documents
    try:
        expansions, elapsed = await service.expand_documents(
            docs, site=site, max_relations=max_relations,
        )
    except Exception as e:
        logger.error("graph_expansion_node failed: %s", e, exc_info=True)
        return {}
    # if no expansions, return empty dict
    if not expansions:
        return {}
    # check what kind of graph expansion (context format) will be used - this is set in config.py - default is inline
    fmt = get_param('graph_context_format', GRAPH_CONTEXT_FORMAT)
    # if boost_inline - inside var docs - put - def apply_apply_graph_boost 
    # this sorts the documents by adding the graph boost score to the reranker score
    # boost_inline: re-rank docs = reranker_score + alpha * graph_boost
    if fmt == "boost_inline":
        docs = _apply_graph_boost(docs, expansions, question, service)
    # written down in the graph_expansion.py file in var on top of the code even though this is async function
    global last_graph_data
    # inside Class GraphExpansionService - 
    # put into var _last_expansions - expansions from the last call to expand_documents - db and hashed url variants
    service._last_expansions = expansions
    # enrich_document_metadata -  new docs with alpha * graph_boost scores added to the metadata
    service.enrich_document_metadata(docs, expansions)
    # build_subgraph_data - build the subgraph data for question, modified docs, and expansions - calls helper function build_subgraph_data
    # calls helper function build_subgraph_data inside Class GraphExpansionService - 
    subgraph = service.build_subgraph_data(question, docs, expansions)
    # save_static_subgraph - save the subgraph data to a static HTML file - calls helper function _save_static_subgraph
    html_path = _save_static_subgraph(question, docs, expansions, service)
    # put the html path into the subgraph data
    subgraph["subgraph_url"] = html_path
    # put the subgraph data into the last_graph_data variable - this is used to display the subgraph in the UI
    last_graph_data = subgraph

    # if no graph expansion, return the documents and the subgraph data
    if fmt == "none":
        return {"_retrieved_docs": docs, "_graph_data": subgraph}
    # get the messages from the state - this is the output from direct_retrieval_node - saved in state['messages']
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        # if inline or boost_inline - inside var new_content - put - def _rebuild_inline
        if fmt in ("inline", "boost_inline"):
            new_content = _rebuild_inline(msg.content, docs, expansions, service)
        # if hint - inside var new_content - put - def _rebuild_hint
        elif fmt == "hint":
            new_content = _rebuild_hint(msg.content, docs, expansions, service, question)
        else:
            graph_block = service.format_graph_context(docs, expansions)
            new_content = msg.content + graph_block if graph_block else msg.content
        # if the new content is the same as the old content, return the documents
        if new_content == msg.content:
            return {"_retrieved_docs": docs}
        # create a new ToolMessage with the new content
        replacement = ToolMessage(
            content=new_content,
            tool_call_id=msg.tool_call_id,
            name=getattr(msg, "name", None),
            id=msg.id,
        )
        # return the new ToolMessage and the documents and the subgraph data
        return {
            "messages": [replacement],
            "_retrieved_docs": docs,
            "_graph_data": subgraph,
        }

    return {}


SUBGRAPH_STATIC_DIR = os.environ.get("SUBGRAPH_STATIC_DIR", "/app/subgraphs")
SUBGRAPH_BASE_URL = os.environ.get("SUBGRAPH_BASE_URL", "/subgraphs")


def _save_static_subgraph(
    question: str,
    docs: list,
    expansions: dict,
    service: GraphExpansionService,
    citation_urls: dict | None = None,
    graph_override: dict | None = None,
) -> str | None:
    """Generate a static HTML subgraph page and return its public URL path."""
    import hashlib, json, html as html_mod
    from pathlib import Path

    if citation_urls is None:
        citation_urls = {}

    import time as _time
    q_hash = hashlib.md5(question.encode()).hexdigest()[:8]
    t_suffix = hex(int(_time.time()))[2:]
    filename = f"{q_hash}_{t_suffix}.html"
    out_dir = Path(SUBGRAPH_STATIC_DIR)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("Cannot create subgraph dir: %s", out_dir)
        return None

    graph = graph_override if graph_override else service.build_subgraph_data(question, docs, expansions)
    nodes_json = json.dumps(graph["nodes"])
    edges_json = json.dumps(graph["edges"])
    q_esc = html_mod.escape(question)
    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])

    colors = GraphExpansionService.SUBGRAPH_COLORS
    legend_types = sorted(set(n["type"] for n in graph["nodes"]))
    legend_html = "".join(
        f'<div class="legend-item"><div class="legend-dot" style="background:{colors.get(lt,"#8d9191")}"></div>{lt}</div>'
        for lt in legend_types
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Subgraph: {q_esc}</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,500;9..40,700&display=swap" rel="stylesheet">
<style>
:root {{ --teal:#00879a; --dark:#28282a; --gray-light:#f4f6f8; --gray-mid:#8d9191; --white:#fff; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
html,body {{ height:100%; overflow:hidden; }}
body {{ font-family:'DM Sans',system-ui,sans-serif; background:var(--gray-light); color:var(--dark); }}
.viz {{ width:100%; height:100%; }}
.legend {{ position:absolute; bottom:0; left:0; right:0; padding:6px 12px; background:rgba(255,255,255,0.9); display:flex; flex-wrap:wrap; gap:4px 12px; border-top:1px solid #e5e7eb; }}
.legend-item {{ display:flex; align-items:center; gap:4px; font-size:10px; color:#5a5a5a; }}
.legend-dot {{ width:7px; height:7px; border-radius:50%; flex-shrink:0; }}
.zoom {{ position:absolute; top:8px; right:8px; z-index:20; display:flex; flex-direction:column; gap:3px; }}
.zbtn {{ width:28px; height:28px; border-radius:6px; border:1px solid #ccd1d4; background:var(--white); color:var(--dark); font-size:15px; cursor:pointer; display:flex; align-items:center; justify-content:center; box-shadow:0 2px 6px rgba(0,0,0,.08); }}
.zbtn:hover {{ border-color:var(--teal); color:var(--teal); }}
.info {{ display:none; position:fixed; bottom:40px; right:8px; background:var(--white); border:1px solid #dde0e3; border-radius:8px; padding:10px; max-width:280px; font-size:11px; z-index:100; box-shadow:0 4px 16px rgba(0,0,0,.12); }}
.info h4 {{ font-size:11px; color:var(--teal); margin-bottom:2px; }}
.info a {{ color:var(--teal); }}
.xbtn {{ position:absolute; top:2px; right:6px; background:none; border:none; color:var(--gray-mid); font-size:13px; cursor:pointer; }}
</style>
</head>
<body>
<div class="viz" id="viz"></div>
<div class="legend">{legend_html}</div>
<div class="zoom">
<button class="zbtn" id="zI">+</button>
<button class="zbtn" id="zO">&minus;</button>
<button class="zbtn" id="zR" style="font-size:11px">&#8634;</button>
</div>
<div class="info" id="info">
<button class="xbtn" onclick="this.parentElement.style.display='none'">&times;</button>
<div id="ic"></div>
</div>
<script>
const N={nodes_json},E={edges_json};
const DIM='#d8dbe0';
function blendColor(hex,alpha){{const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);const bg=244;const mix=c=>Math.round(c*alpha+bg*(1-alpha));return'#'+[mix(r),mix(g),mix(b)].map(c=>c.toString(16).padStart(2,'0')).join('');}}
const ds=new vis.DataSet(N.map(n=>{{const h=n.hop3?3:(n.type==='Question'||n.type==='Source'?0:1);const isQ=n.type==='Question';let bg,border,fc,fsz,bw;if(h===0){{bg=n.color;border=n.color;fc='#000';fsz=isQ?15:13;bw=isQ?2:(n.expanded?3:0);}}else if(h===1){{bg=blendColor(n.color,0.7);border=blendColor(n.color,0.5);fc='#333';fsz=11;bw=1;}}else{{bg=blendColor(n.color,0.3);border=blendColor(n.color,0.2);fc=blendColor('#28282a',0.3);fsz=8;bw=0;}}const m=isQ?5:1;const sh=h===0;const ret={{id:n.id,label:n.label,color:{{background:isQ?'#1a1a1a':bg,border:isQ?'#000':border,highlight:{{background:n.color,border:n.color}}}},size:n.size,shape:isQ?'diamond':'dot',font:{{color:'#000',size:fsz,bold:isQ,strokeWidth:3,strokeColor:'#fff'}},widthConstraint:{{maximum:isQ?180:(h===0?150:120)}},borderWidth:isQ?3:bw,mass:m,shadow:sh?{{enabled:true,color:'rgba(0,0,0,0.15)',size:12,x:2,y:3}}:false,scaling:{{label:{{enabled:isQ,min:14,max:18}}}},_raw:n}};if(isQ){{ret.x=0;ret.y=0;ret.fixed={{x:true,y:true}};}}return ret;}}));
const es=new vis.DataSet(E.map((e,i)=>{{const h=e.hop||1;const eOp=h<=1?0.6:h===2?0.3:0.15;const eCol=h<=1?'#c8ced6':h===2?'#dde0e3':'#e8eaed';return{{id:'e'+i,from:e.from,to:e.to,label:h<=1?(e.label||''):'',font:{{size:h<=1?9:7,color:h<=1?'#555':'#aab0b5',strokeWidth:0}},color:{{color:eCol,opacity:eOp}},width:h<=1?1.5:0.5,dashes:h>=3?[3,4]:false,arrows:{{to:{{enabled:true,scaleFactor:h<=1?0.3:0.2}}}},smooth:{{type:'continuous'}}}};}}));
const nC=N.length;const grav=nC>40?-100:-60;const sLen=nC>40?140:100;
const net=new vis.Network(document.getElementById('viz'),{{nodes:ds,edges:es}},{{physics:{{solver:'forceAtlas2Based',forceAtlas2Based:{{gravitationalConstant:grav,centralGravity:0.02,springLength:sLen,springConstant:0.02,damping:0.4,avoidOverlap:0.5}},stabilization:{{iterations:400,fit:true}},maxVelocity:30}},interaction:{{hover:true,zoomView:false,dragView:true}}}});
const coreIds=N.filter(n=>n.type==='Question'||n.type==='Source').map(n=>n.id);
net.once('stabilizationIterationsDone',()=>{{net.setOptions({{physics:false}});if(coreIds.length){{net.fit({{nodes:coreIds,animation:{{duration:400}}}});}}else{{net.fit();}}}});
const origN=new Map();ds.forEach(n=>origN.set(n.id,{{bg:n.color.background,border:n.color.border,fc:n.font?.color||'#000'}}));
const origE=new Map();es.forEach(e=>origE.set(e.id,{{color:e.color?.color||'#c8ced6',op:e.color?.opacity||0.6}}));
net.on('click',p=>{{if(p.nodes.length){{const nid=p.nodes[0];const cn=new Set(net.getConnectedNodes(nid));cn.add(nid);ds.forEach(n=>{{const o=origN.get(n.id);ds.update({{id:n.id,color:{{background:cn.has(n.id)?o.bg:DIM,border:cn.has(n.id)?o.border:DIM}},font:{{color:cn.has(n.id)?o.fc:'#c0c4ca'}}}});}});es.forEach(e=>{{const on=cn.has(e.from)&&cn.has(e.to);const o=origE.get(e.id);es.update({{id:e.id,color:{{color:on?'#90999f':'#e8eaed',opacity:on?0.7:0.1}}}});}});const r=ds.get(nid)._raw;let h='<h4>'+r.label+'</h4><p>Type: '+r.type+'</p>';if(r.url)h+='<p><a href="'+r.url+'" target="_blank">Open &nearr;</a></p>';document.getElementById('ic').innerHTML=h;document.getElementById('info').style.display='block';}}else{{ds.forEach(n=>{{const o=origN.get(n.id);ds.update({{id:n.id,color:{{background:o.bg,border:o.border}},font:{{color:o.fc}}}});}});es.forEach(e=>{{const o=origE.get(e.id);es.update({{id:e.id,color:{{color:o.color,opacity:o.op}}}});}});document.getElementById('info').style.display='none';}}}});
net.on('doubleClick',p=>{{if(p.nodes.length){{const r=ds.get(p.nodes[0])._raw;if(r.url)window.open(r.url,'_blank');}}}});
document.getElementById('zI').onclick=()=>net.moveTo({{scale:Math.min(net.getScale()*1.3,2),animation:{{duration:200}}}});
document.getElementById('zO').onclick=()=>net.moveTo({{scale:Math.max(net.getScale()/1.3,0.05),animation:{{duration:200}}}});
document.getElementById('zR').onclick=()=>net.fit({{animation:{{duration:400}}}});
</script>
</body>
</html>"""

    out_path = out_dir / filename
    out_path.write_text(html, encoding="utf-8")
    logger.info("Saved static subgraph: %s (%d nodes, %d edges)", out_path, n_nodes, n_edges)
    return f"{SUBGRAPH_BASE_URL}/{filename}"

# context: used in graph_expansion_node.py - when called by boost_inline - apply graph boost to the documents
# description: re-rank docs using graph connectivity boost scores.
# input given by: list of documents - docs - this is the output from async expand_documents function in graph_expansion.py
# - expansions - this is the output from async expand_documents function in graph_expansion.py - neo4j db and hashed url variants
# - question - this is the user question - saved in state['_question']
# - service - this is the instance of the graph expansion service - saved in service variable
# return: list - this is the re-ranked documents
def _apply_graph_boost(
    docs: list,
    expansions: dict,
    question: str,
    service: GraphExpansionService,
) -> list:
    """Re-rank docs using graph connectivity boost scores."""
    boosts = service.compute_graph_boost_scores(
        docs, expansions, question, alpha=1.0,
    )
    # get the alpha from the config - this is neo4j config - default is 0.15 - which is favored graph or other reranker scores.
    alpha = GRAPH_BOOST_ALPHA

    scored = []
    # loop through the documents and re-rank them using the graph boost scores
    for i, doc in enumerate(docs):
    # inside var base - get reranker score inside metadata - this is the score from the reranker - saved in doc.metadata['_reranker_score']
        base = doc.metadata.get(
            "_reranker_score",
            doc.metadata.get("_final_score", 1.0 - i * 0.01),
        )
    # add the graph boost score to the reranker score  by multiplying the graph boost score by the alpha and adding it to the base score
        scored.append((base + alpha * boosts[i], doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    # sort the documents by the score now changed to use the graph boost score also
    reordered = [doc for _, doc in scored]
    # count the number of documents that have changed by the graph boost score
    n_changed = sum(1 for a, b in zip(docs, reordered) if a is not b)
    if n_changed:
        logger.info("graph_boost: re-ranked %d/%d docs (alpha=%.2f)", n_changed, len(docs), alpha)
    # return the re-ranked documents
    return reordered


def _rebuild_inline(
    original_content: str,
    docs: list,
    expansions: dict,
    service: GraphExpansionService,
) -> str:
    """Split ToolMessage into per-source blocks and append [Graph: ...] to each."""
    blocks = original_content.split(SOURCE_SEPARATOR)
    enriched = []
    for i, block in enumerate(blocks):
        if i < len(docs):
            graph_line = service.format_inline_source_context(docs[i], expansions)
            if graph_line:
                block = block + "\n" + graph_line
        enriched.append(block)
    return SOURCE_SEPARATOR.join(enriched)


def _rebuild_hint(
    original_content: str,
    docs: list,
    expansions: dict,
    service: GraphExpansionService,
    question: str,
) -> str:
    """Split ToolMessage into per-source blocks and append conditional hints."""
    blocks = original_content.split(SOURCE_SEPARATOR)
    enriched = []
    for i, block in enumerate(blocks):
        if i < len(docs):
            hint = service.format_conditional_hint(docs[i], expansions, question)
            if hint:
                block = block + "\n" + hint
        enriched.append(block)
    return SOURCE_SEPARATOR.join(enriched)
