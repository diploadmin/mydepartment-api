"""Read-only viewer for ingested documents that have no public source_url.

Uploaded files (course PDFs, markdown dumps) are stored in Weaviate without a
``source_url``, so their source cards had nowhere to link. This route renders
``Documents.full_text`` and highlights the cited passage, using the same
``diplo-deep-link-text`` payload the PDF proxy consumes.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from weaviate.classes.query import Filter

from app.ai.ai_services.retrievers.utils import resolve_collection_name
from app.core.collections import DOCUMENT
from app.core.logging import logger
from app.core.singleton import Singleton
from app.core.weaviate_props import (
    COPYRIGHT,
    DOCUMENT_HASH,
    FULL_TEXT,
    PARENT_DOCUMENT_HASH,
    PUBLISH_DATE,
    doc_title,
)

router = APIRouter()

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  body {
    margin: 0;
    background: #f6f7f9;
    color: #1a1c1f;
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  header {
    position: sticky; top: 0; z-index: 2;
    padding: 14px 20px;
    background: #11314f; color: #fff;
    box-shadow: 0 1px 6px rgba(0,0,0,.25);
  }
  header h1 { margin: 0; font-size: 17px; font-weight: 600; }
  header p { margin: 4px 0 0; font-size: 12px; opacity: .75; }
  main {
    max-width: 860px; margin: 24px auto 80px; padding: 32px 36px;
    background: #fff; border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,.12);
  }
  #doc { white-space: pre-wrap; word-wrap: break-word; }
  mark {
    background: #ffe9a8; color: inherit;
    padding: 1px 0; border-radius: 2px;
    box-shadow: 0 0 0 2px #ffe9a8;
  }
  mark.diplo-primary { background: #ffd257; box-shadow: 0 0 0 3px #ffd257; }
  #nohit { display: none; margin: 0 0 18px; padding: 10px 14px;
           background: #fff4d6; border-left: 4px solid #e9b949; font-size: 13px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p>__SUBTITLE__</p>
</header>
<main>
  <p id="nohit">The cited passage could not be located in this document, so nothing is highlighted.</p>
  <div id="doc">__BODY__</div>
</main>
<script>
(function () {
  // Highlight params arrive in the hash (PDF proxy convention) or query string.
  function readParam(name) {
    var hash = window.location.hash.replace(/^#/, '');
    var sources = [hash, window.location.search.replace(/^\\?/, '')];
    for (var i = 0; i < sources.length; i++) {
      var parts = sources[i].split('&');
      for (var j = 0; j < parts.length; j++) {
        var eq = parts[j].indexOf('=');
        if (eq > 0 && decodeURIComponent(parts[j].slice(0, eq)) === name) {
          return decodeURIComponent(parts[j].slice(eq + 1).replace(/\\+/g, ' '));
        }
      }
    }
    return '';
  }

  var payload = readParam('diplo-deep-link-text');
  if (!payload) return;

  // Payload layout: "<paragraph>===SENTENCE===<sent>|||SENT|||<sent>..."
  var chunks = payload.split('===SENTENCE===');
  var needles = [];
  if (chunks.length > 1) {
    needles = chunks[1].split('|||SENT|||');
  } else {
    needles = [chunks[0]];
  }
  needles = needles.map(function (s) { return s.trim(); }).filter(Boolean);
  if (!needles.length) return;

  var el = document.getElementById('doc');
  var original = el.textContent;

  // Collapse whitespace and lowercase, keeping an index map back to `original`
  // so PDF-extracted line breaks do not defeat matching.
  function normalize(text) {
    var out = '', map = [], prevSpace = false;
    for (var i = 0; i < text.length; i++) {
      var ch = text[i];
      if (/\\s/.test(ch)) {
        if (prevSpace || !out) continue;
        out += ' '; map.push(i); prevSpace = true;
      } else {
        out += ch.toLowerCase(); map.push(i); prevSpace = false;
      }
    }
    return { text: out, map: map };
  }

  var doc = normalize(original);
  var ranges = [];

  needles.forEach(function (needle, order) {
    var n = normalize(needle).text.trim();
    if (n.length < 12) return;
    var at = doc.text.indexOf(n);
    if (at === -1) {
      // Extraction artifacts (hyphenation, stray spaces) — settle for a prefix.
      var probe = n.slice(0, Math.max(24, Math.floor(n.length * 0.5)));
      at = doc.text.indexOf(probe);
      if (at === -1) return;
      n = probe;
    }
    ranges.push({
      start: doc.map[at],
      end: doc.map[at + n.length - 1] + 1,
      primary: order === 0
    });
  });

  if (!ranges.length) {
    document.getElementById('nohit').style.display = 'block';
    return;
  }

  ranges.sort(function (a, b) { return a.start - b.start; });
  var merged = [ranges[0]];
  for (var i = 1; i < ranges.length; i++) {
    var last = merged[merged.length - 1];
    if (ranges[i].start <= last.end) {
      last.end = Math.max(last.end, ranges[i].end);
      last.primary = last.primary || ranges[i].primary;
    } else {
      merged.push(ranges[i]);
    }
  }

  var frag = document.createDocumentFragment();
  var cursor = 0;
  merged.forEach(function (r) {
    if (r.start > cursor) {
      frag.appendChild(document.createTextNode(original.slice(cursor, r.start)));
    }
    var mark = document.createElement('mark');
    if (r.primary) mark.className = 'diplo-primary';
    mark.textContent = original.slice(r.start, r.end);
    frag.appendChild(mark);
    cursor = r.end;
  });
  if (cursor < original.length) {
    frag.appendChild(document.createTextNode(original.slice(cursor)));
  }
  el.textContent = '';
  el.appendChild(frag);

  var target = el.querySelector('mark.diplo-primary') || el.querySelector('mark');
  if (target) {
    target.scrollIntoView({ block: 'center' });
  }
})();
</script>
</body>
</html>
"""


def _fetch_document(document_hash: str):
    client = Singleton().weaviate_client
    collection = client.collections.get(resolve_collection_name(DOCUMENT))
    for prop in (DOCUMENT_HASH, PARENT_DOCUMENT_HASH):
        result = collection.query.fetch_objects(
            limit=1,
            filters=Filter.by_property(prop).equal(document_hash),
        )
        if result.objects:
            return result.objects[0].properties or {}
    return None


@router.get(
    "/{document_hash}",
    response_class=HTMLResponse,
    summary="Render an ingested document that has no public source URL",
)
async def view_document(document_hash: str) -> HTMLResponse:
    document_hash = (document_hash or "").strip().lower()
    if not document_hash.isalnum() or not 16 <= len(document_hash) <= 64:
        raise HTTPException(status_code=400, detail="Invalid document hash")

    try:
        props = _fetch_document(document_hash)
    except Exception as exc:
        logger.error(f"[doc-view] lookup failed for {document_hash}: {exc}")
        raise HTTPException(status_code=503, detail="Document store unavailable") from exc

    if not props:
        raise HTTPException(status_code=404, detail="Document not found")

    # Copyrighted material is excluded from retrieval, so never serve it here.
    if int(props.get(COPYRIGHT) or 0) == 1:
        raise HTTPException(status_code=403, detail="Document is not viewable")

    body = str(props.get(FULL_TEXT) or "").strip()
    if not body:
        raise HTTPException(status_code=404, detail="Document has no stored text")

    title = doc_title(props) or "Document"
    pub = props.get(PUBLISH_DATE)
    subtitle_parts = ["Stored document — no public source URL"]
    if pub:
        subtitle_parts.append(str(pub)[:10])

    from html import escape, unescape

    page = (
        _PAGE_TEMPLATE
        .replace("__TITLE__", escape(title))
        .replace("__SUBTITLE__", escape(" · ".join(subtitle_parts)))
        .replace("__BODY__", escape(unescape(body)))
    )
    return HTMLResponse(content=page)
