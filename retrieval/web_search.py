"""
Online Web & Academic Search + Autonomous Crystallization Engine.
Enables the assistant to:
1. Detect online/offline network status seamlessly.
2. Query live academic literature (arXiv Atom API, Wikipedia API, DuckDuckGo).
3. Synthesize the findings citing [W1]..[Wk].
4. Automatically crystallize the synthesized knowledge into Obsidian Vault
   (Vault/02_Theory_Notes/Web_{slug}.md) and upsert into ChromaDB ('paper_chunks').
5. Enable 100% offline retrieval of crystallized knowledge on future queries!
"""

import sys
import re
import socket
import hashlib
import urllib.parse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import requests
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

ARXIV_API_URL = "http://export.arxiv.org/api/query"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = "DeepLearningLabAssistant/2.0 (Academic Research; Autonomous Knowledge Harvester)"
HEADERS = {"User-Agent": DEFAULT_USER_AGENT}


def is_online(timeout: float = 1.5) -> bool:
    """
    Ultra-fast socket probe to verify real internet connectivity.
    Gracefully handles sandbox or local offline execution.
    """
    try:
        s = socket.create_connection(("google.com", 80), timeout=timeout)
        s.close()
        return True
    except Exception:
        try:
            r = requests.get("https://1.1.1.1", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False


class AcademicWebSearcher:
    """Multi-source live academic web searcher and auto-crystallizer."""

    def __init__(self, timeout: float = 3.5):
        self.timeout = timeout

    def search_arxiv(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Queries arXiv REST API for peer-reviewed preprints and research papers."""
        clean_q = re.sub(r'[^a-zA-Z0-9\s_-]', ' ', query)
        terms = [t for t in clean_q.split() if len(t) > 2][:6]
        if not terms:
            terms = ["deep", "learning"]

        search_query = "+AND+".join([f"all:{t}" for t in terms])
        url = f"{ARXIV_API_URL}?search_query={search_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=self.timeout)
            if resp.status_code != 200:
                return []

            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            results = []

            for entry in root.findall("atom:entry", ns):
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                id_elem = entry.find("atom:id", ns)
                published_elem = entry.find("atom:published", ns)

                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "Untitled"
                summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
                url_link = id_elem.text.strip() if id_elem is not None else ""
                published = published_elem.text[:10] if published_elem is not None else ""

                authors = []
                for a in entry.findall("atom:author", ns):
                    name_elem = a.find("atom:name", ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())

                if len(summary) > 450:
                    summary = summary[:450] + "..."

                results.append({
                    "title": title,
                    "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
                    "summary": summary,
                    "url": url_link,
                    "published": published,
                    "source": "arXiv Live"
                })
            return results
        except Exception:
            return []

    def search_wikipedia(self, query: str, max_results: int = 2) -> List[Dict[str, Any]]:
        """Queries Wikipedia API for formal foundational concepts, theory, and mathematical definitions."""
        try:
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "utf8": "1",
                "format": "json",
                "srlimit": max_results
            }
            resp = requests.get(WIKIPEDIA_API_URL, params=params, headers=HEADERS, timeout=self.timeout)
            if resp.status_code != 200:
                return []

            data = resp.json()
            items = data.get("query", {}).get("search", [])
            results = []
            for item in items:
                title = item.get("title", "")
                raw_snippet = item.get("snippet", "")
                clean_snippet = re.sub(r'<[^>]+>', '', raw_snippet).strip()
                page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

                results.append({
                    "title": f"Wikipedia: {title}",
                    "authors": "Wikipedia Contributors",
                    "summary": clean_snippet,
                    "url": page_url,
                    "published": datetime.now().strftime("%Y-%m"),
                    "source": "Wikipedia Academic"
                })
            return results
        except Exception:
            return []

    def search_duckduckgo_lite(self, query: str, max_results: int = 2) -> List[Dict[str, Any]]:
        """Lightweight search for technical implementations, benchmarks, and GitHub repositories."""
        try:
            resp = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                timeout=self.timeout
            )
            if resp.status_code != 200:
                return []

            snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
            titles = re.findall(r'<a class="result__url"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', resp.text, re.DOTALL)

            results = []
            for i in range(min(len(snippets), max_results)):
                raw_snip = re.sub(r'<[^>]+>', '', snippets[i]).strip()
                raw_url = titles[i][0].strip() if i < len(titles) else ""
                if raw_url.startswith("//duckduckgo.com/l/?uddg="):
                    m = re.search(r'uddg=([^&]+)', raw_url)
                    if m:
                        raw_url = urllib.parse.unquote(m.group(1))

                results.append({
                    "title": f"Web Referansı #{i+1}",
                    "authors": "Web/GitHub",
                    "summary": raw_snip,
                    "url": raw_url,
                    "published": datetime.now().strftime("%Y-%m"),
                    "source": "Web Live"
                })
            return results
        except Exception:
            return []

    def search_academic_web(self, query: str, max_results: int = 4) -> List[Dict[str, Any]]:
        """Orchestrates multi-source academic retrieval across arXiv, Wikipedia, and the open web."""
        results = []
        # 1. arXiv Preprints
        arxiv_res = self.search_arxiv(query, max_results=2)
        results.extend(arxiv_res)

        # 2. Wikipedia Concept Definition
        wiki_res = self.search_wikipedia(query, max_results=2)
        results.extend(wiki_res)

        # 3. DuckDuckGo / Open Web if insufficient
        if len(results) < max_results:
            ddg_res = self.search_duckduckgo_lite(f"{query} deep learning", max_results=max_results - len(results))
            results.extend(ddg_res)

        return results[:max_results]

    def get_live_web_context(self, query: str) -> Dict[str, Any]:
        """
        Combines live arXiv papers, Wikipedia concepts, and technical web links
        into a unified Markdown context block labeled [W1]..[W3].
        """
        if not is_online():
            return {"has_web": False, "packed_context": "", "sources": [], "raw_results": []}

        raw_results = self.search_academic_web(query, max_results=3)
        if not raw_results:
            return {"has_web": False, "packed_context": "", "sources": [], "raw_results": []}

        packed_context = self.pack_web_context(raw_results)
        sources = []
        for idx, item in enumerate(raw_results):
            sources.append({
                "tag": f"W{idx+1}",
                "title": item.get("title", "Online Source"),
                "url": item.get("url", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", "Web")
            })

        return {
            "has_web": True,
            "packed_context": packed_context,
            "sources": sources,
            "raw_results": raw_results
        }

    def pack_web_context(self, results: List[Dict[str, Any]]) -> str:
        """Packs live web search results into [W1]..[Wk] citation context."""
        if not results:
            return ""

        lines = ["### [CANLI İNTERNET & AKADEMİK LİTERATÜR BULGULARI]:"]
        for idx, item in enumerate(results):
            tag = f"W{idx+1}"
            title = item.get("title", "Online Source")
            url = item.get("url", "")
            summary = item.get("summary", "")
            authors = item.get("authors", "")
            pub = item.get("published", "")
            auth_info = f" ({authors}, {pub})" if authors else ""

            lines.append(f"[{tag}] **{title}**{auth_info}\nBağlantı: {url}\nÖzet/Bulgular: {summary}\n")

        return "\n".join(lines)

    def crystalize_knowledge(
        self,
        query: str,
        synthesized_answer: str,
        web_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        AUTONOMOUS KNOWLEDGE CRYSTALLIZATION:
        1. Generates an Obsidian Markdown Note in Vault/02_Theory_Notes/Web_{slug}.md.
        2. Embeds & indexes the synthesized knowledge chunk into ChromaDB 'paper_chunks'.
        3. Clears searcher BM25 cache so the new knowledge is immediately discoverable offline!
        """
        try:
            clean_q = re.sub(r'[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\s_-]', '', query).strip()
            slug = "_".join(clean_q.split()[:5]).capitalize() or "Research_Note"
            filename = f"Web_{slug}.md"
            target_dir = config.VAULT_DIR / "02_Theory_Notes"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename

            # Deduplicate filename if exists
            counter = 1
            while target_path.exists():
                filename = f"Web_{slug}_{counter}.md"
                target_path = target_dir / filename
                counter += 1

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Format references
            refs_md = []
            yaml_sources = []
            for idx, r in enumerate(web_results):
                tag = f"W{idx+1}"
                t = r.get("title", "Kaynak")
                u = r.get("url", "")
                s = r.get("source", "Web")
                refs_md.append(f"- **`[{tag}]`** [{t}]({u}) *({s})*")
                yaml_sources.append(f"  - title: \"{t}\"\n    url: \"{u}\"")

            refs_block = "\n".join(refs_md) if refs_md else "- *Doğrudan web sentezi*"
            yaml_block = "\n".join(yaml_sources) if yaml_sources else "  - title: Web Research"

            # Clean synthesized answer from any trailing footer
            clean_body = synthesized_answer.split("---")[0].strip()

            obsidian_content = f"""---
title: "{clean_q}"
type: web_crystallized_note
date: {now_str}
tags:
  - web-learning
  - theory
  - auto-crystallized
sources:
{yaml_block}
---

# 🌐 {clean_q}

> [!NOTE] 🧠 Otonom Canlı Web Araştırması & Kristalize Bilgi
> Bu not, internet bağlantısı aktifken harici akademik kaynaklardan (arXiv, Wikipedia, teknik web) taranarak yapay zeka tarafından sentezlenmiş ve yerel hafızaya kalıcı olarak eklenmiştir. 
> **Artık internet bağlantısı kesilse dahi sistem bu bilgiyi yerel ChromaDB hafızasından çağırabilir.**

## 📚 İncelenen Canlı Kaynaklar
{refs_block}

## 🔬 Sentezlenmiş Teknik İçerik
{clean_body}

---
*İlgili Notlar:* [[00_Index]], [[Web_Knowledge_Base]], [[Derin Öğrenme Teorisi]]
"""
            target_path.write_text(obsidian_content, encoding="utf-8")

            # 2. ChromaDB Upsert into 'paper_chunks'
            from ingest.indexer import Indexer
            indexer = Indexer()
            doc_id = f"web_{hashlib.md5(clean_q.encode('utf-8')).hexdigest()[:12]}"

            doc_text = f"Title: {clean_q}\nSource: Web Academic Research\n\n{clean_body}"
            metadata = {
                "title": f"Web: {clean_q[:80]}",
                "section_name": "Web Synthesis",
                "paper_id": f"web_{slug.lower()}",
                "type": "web_crystallized",
                "url": web_results[0].get("url", "") if web_results else "",
                "created_at": now_str
            }

            indexer.collections["paper_chunks"].upsert(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[metadata]
            )

            # 3. Invalidate BM25 cache on hybrid searcher
            try:
                from retrieval.search import searcher
                searcher._bm25_cache.clear()
            except Exception:
                pass

            return {
                "success": True,
                "filename": filename,
                "filepath": str(target_path),
                "doc_id": doc_id
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Global singleton instance
web_searcher = AcademicWebSearcher()
online_searcher = web_searcher
online_searcher = web_searcher
