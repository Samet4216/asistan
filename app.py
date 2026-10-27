"""
Local Deep Learning Lab Assistant — Interactive Rich CLI (Faz 9).
Provides a dedicated terminal interface with slash commands,
GPU thermal monitoring, git context, telemetry autopsy, and Obsidian memory integration.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe Windows stdout configuration
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

import config
import llm
import lang_guard
from utils import get_gpu_status
from retrieval.git_context import get_git_diff
from ingest.log_parser import LogParser
from answer.pipeline import pipeline
from memory.approval import approval_manager
from answer.shape_check import dry_run_module
from answer.challenge import challenger


console = Console()


def print_banner():
    """Prints welcoming banner with live GPU status."""
    gpu = get_gpu_status()
    gpu_info = (
        f"[green]{gpu['name']}[/] | VRAM: [bold]{gpu['used_mb']} / {gpu['total_mb']} MiB[/] | "
        f"Temp: [bold]{gpu['temp_c']}°C[/] | GPU Load: [bold]{gpu['util_pct']}%[/]"
        if gpu["available"] else "[yellow]GPU Telemetry Unavailable[/]"
    )

    banner_text = (
        f"[bold cyan]Local Deep Learning Lab Assistant (v2)[/bold cyan]\n"
        f"[dim]Hardware profile: ASUS TUF A15 | RTX 4050 6GB Laptop GPU | 100% Technical English[/dim]\n\n"
        f"• Model: [bold green]{config.MODEL}[/] (Ollama CUDA)\n"
        f"• Embeddings: [bold green]{config.EMBED_MODEL}[/] (CPU 768-dim)\n"
        f"• Hardware: {gpu_info}\n"
        f"• Obsidian Vault: [dim]{config.VAULT_DIR}[/dim]\n\n"
        f"[dim]Type your question or enter [bold cyan]/help[/bold cyan] to see available commands.[/dim]"
    )
    console.print(Panel(banner_text, title="🔬 Lab Assistant", border_style="cyan", box=box.ROUNDED))


def print_help():
    """Renders command cheat sheet."""
    table = Table(title="Available Slash Commands", box=box.SIMPLE_HEAVY)
    table.add_column("Command", style="cyan bold", width=22)
    table.add_column("Description", style="white")

    table.add_row("/diff", "Inspect uncommitted git code diff")
    table.add_row("/logs <path>", "Run log autopsy on training results (e.g., results.csv)")
    table.add_row("/drafts", "List unapproved notes in Vault/Drafts/")
    table.add_row("/save <title>", "Save last answer as a draft note to Vault/Drafts/")
    table.add_row("/approve <id>", "Approve a draft into permanent Obsidian Vault & ChromaDB")
    table.add_row("/reject <id>", "Discard a draft from Vault/Drafts/")
    table.add_row("/tr", "Translate last answer into academic Turkish (preserves math)")
    table.add_row("/dryrun <path>", "Dry-run PyTorch module on CPU: check shapes, backward pass, & FLOPs")
    table.add_row("/challenge <hypo>", "Devil's Advocate mode: stress-test hypotheses against failure modes & GPU limits")
    table.add_row("/explain <topic>", "Deep-dive tutorial mode: comprehensive pedagogical lecture (math, tensor flow & code)")
    table.add_row("/gpu  or  /vram", "Show real-time RTX 4050 VRAM usage and temperature")
    table.add_row("/unload", "Unload model weights from VRAM to cool GPU")
    table.add_row("/exit  or  /quit", "Unload weights and exit")
    console.print(table)


def run_repl():
    """Main interactive REPL loop."""
    print_banner()
    last_answer = ""
    last_query = ""

    while True:
        try:
            user_input = console.input("\n[bold cyan]assistant > [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow][*] Exiting... Unloading model from VRAM.[/yellow]")
            llm.unload()
            break

        if not user_input:
            continue

        cmd = user_input.split()[0].lower()

        if cmd in ("/exit", "/quit", "exit", "quit"):
            console.print("[yellow][*] Unloading model weights from VRAM...[/yellow]")
            llm.unload()
            console.print("[green]✓ VRAM freed. Goodbye![/green]")
            break

        elif cmd == "/help":
            print_help()

        elif cmd in ("/gpu", "/vram"):
            gpu = get_gpu_status()
            if gpu["available"]:
                t = Table(box=box.ROUNDED, title="GPU Status")
                t.add_column("Property", style="cyan")
                t.add_column("Value", style="green bold")
                t.add_row("Device", gpu["name"])
                t.add_row("VRAM Used", f"{gpu['used_mb']} MiB")
                t.add_row("VRAM Free", f"{gpu['free_mb']} MiB")
                t.add_row("VRAM Total", f"{gpu['total_mb']} MiB")
                t.add_row("Temperature", f"{gpu['temp_c']} °C")
                t.add_row("Core Utilization", f"{gpu['util_pct']} %")
                idle = llm.get_idle_status()
                if idle.get("is_loaded"):
                    rem_m = idle["remaining_seconds"] // 60
                    rem_s = idle["remaining_seconds"] % 60
                    t.add_row("Auto-Unload Timer", f"Active ({rem_m}m {rem_s}s remaining / {idle['timeout_seconds']//60}m idle limit)")
                else:
                    t.add_row("Auto-Unload Timer", f"Idle / Model unloaded (10m watchdog active)")
                console.print(t)
            else:
                console.print("[red][!] Unable to query nvidia-smi.[/red]")

        elif cmd == "/unload":
            console.print("[yellow][*] Unloading model from GPU VRAM...[/yellow]")
            msg = llm.unload()
            console.print(f"[green]✓ {msg}[/green]")

        elif cmd == "/diff":
            with console.status("[cyan]Reading local git changes...[/cyan]"):
                diff_text = get_git_diff()
            console.print(Panel(diff_text, title="Local Git Working Tree Diff", border_style="magenta", box=box.ROUNDED))

        elif cmd == "/logs":
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[yellow]Usage: /logs <path_to_run_dir>[/yellow]")
                continue
            run_path = Path(parts[1].strip('"\''))
            if not run_path.exists():
                console.print(f"[red][!] Directory not found: {run_path}[/red]")
                continue

            with console.status("[cyan]Analyzing training telemetry...[/cyan]"):
                parser = LogParser(run_path)
                report = parser.analyze_results()

            symptoms = "\n".join(f"• {s}" for s in report.get("symptoms", [])) if report.get("symptoms") else "None detected."
            diag_text = (
                f"**Status:** {report.get('status')}\n\n"
                f"**Epochs Analyzed:** {report.get('epochs_analyzed')}\n\n"
                f"**Symptoms Detected:**\n{symptoms}\n\n"
                f"**Summary Diagnosis:**\n{report.get('diagnosis')}"
            )
            console.print(Panel(Markdown(diag_text), title="Telemetry Autopsy Report", border_style="red" if report.get("status") == "FAIL" else "green", box=box.ROUNDED))

        elif cmd == "/drafts":
            drafts = approval_manager.list_drafts()
            if not drafts:
                console.print("[dim]No pending drafts in Vault/Drafts/.[/dim]")
            else:
                t = Table(title=f"Pending Drafts ({len(drafts)})", box=box.SIMPLE_HEAVY)
                t.add_column("Draft ID", style="cyan")
                t.add_column("Title", style="white bold")
                t.add_column("Category", style="green")
                t.add_column("Created", style="dim")
                for d in drafts:
                    t.add_row(d["id"], d["title"], d["category"], d.get("created_at", "")[:16])
                console.print(t)

        elif cmd == "/save":
            if not last_answer:
                console.print("[yellow][!] No previous answer to save. Ask a question first.[/yellow]")
                continue
            parts = user_input.split(maxsplit=1)
            title = parts[1] if len(parts) > 1 else (last_query[:40] or "Lab_Note")
            draft_path = approval_manager.create_draft(
                title=title,
                content=f"**Query:** {last_query}\n\n**Answer:**\n{last_answer}",
                category="Theory_Notes",
                tags=["session-save"]
            )
            console.print(f"[green]✓ Draft created at: [bold]{draft_path.name}[/bold] in Vault/Drafts/[/green]")
            console.print(f"[dim]Use '/approve {draft_path.stem}' to move to permanent vault.[/dim]")

        elif cmd == "/approve":
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[yellow]Usage: /approve <draft_id>[/yellow]")
                continue
            draft_id = parts[1].strip()
            with console.status("[cyan]Approving and indexing into Obsidian Vault...[/cyan]"):
                approved_path = approval_manager.approve_draft(draft_id)
            if approved_path:
                console.print(f"[green]✓ Approved note moved to: [bold]{approved_path}[/bold][/green]")
                console.print("[green]✓ Indexed into ChromaDB and committed to Vault Git.[/green]")
            else:
                console.print(f"[red][!] Draft not found: {draft_id}[/red]")

        elif cmd == "/reject":
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[yellow]Usage: /reject <draft_id>[/yellow]")
                continue
            draft_id = parts[1].strip()
            if approval_manager.reject_draft(draft_id):
                console.print(f"[green]✓ Rejected and removed draft '{draft_id}'.[/green]")
            else:
                console.print(f"[red][!] Draft not found: {draft_id}[/red]")

        elif cmd == "/tr":
            if not last_answer:
                console.print("[yellow][!] No previous answer to translate. Ask a question first.[/yellow]")
                continue
            with console.status("[cyan]Translating technical response to Turkish...[/cyan]"):
                tr_text = lang_guard.translate_to_turkish(last_answer)
            console.print(Panel(Markdown(tr_text), title="🇹🇷 Academic Turkish Translation", border_style="blue", box=box.ROUNDED))

        elif cmd == "/dryrun":
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[yellow]Usage: /dryrun <path_to_python_file>[/yellow]")
                continue
            src_path = Path(parts[1].strip('"\''))
            if not src_path.exists():
                console.print(f"[red][!] File not found: {src_path}[/red]")
                continue

            with console.status("[cyan]Verifying AST security and executing CPU tensor dry-run...[/cyan]"):
                code_text = src_path.read_text(encoding="utf-8")
                res = dry_run_module(code_text)

            if res.get("passed"):
                t = Table(box=box.ROUNDED, title=f"✓ Module Verified: {res.get('module_name')}")
                t.add_column("Check", style="cyan")
                t.add_column("Result", style="green bold")
                t.add_row("Input Shape", str(res.get("input_shape")))
                t.add_row("Output Shape", str(res.get("output_shape")))
                t.add_row("Backward Autograd Pass", "PASSED" if res.get("backward_passed") else "FAILED")
                t.add_row("Trainable Parameters", f"{res.get('trainable_params'):,}")
                t.add_row("Device", "CPU (0 MB GPU VRAM used)")
                console.print(t)
            else:
                err_text = (
                    f"**Failure Stage:** {res.get('error_stage')}\n\n"
                    f"**Error Details:**\n{res.get('error_message')}"
                )
                console.print(Panel(Markdown(err_text), title="[red]✗ Tensor Dry-Run Failed[/red]", border_style="red", box=box.ROUNDED))

        elif cmd == "/challenge":
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[yellow]Usage: /challenge <your_architectural_hypothesis>[/yellow]")
                continue
            hypothesis = parts[1].strip()

            with console.status("[cyan]Evaluating hypothesis against failure modes and GPU limits (Devil's Advocate)...[/cyan]"):
                res = challenger.challenge_hypothesis(hypothesis)

            verdict = res.get("verdict", "HIGH_RISK")
            v_color = "red" if verdict == "REJECT" else ("yellow" if verdict == "HIGH_RISK" else "green")
            
            critique_lines = []
            for c in res.get("criticisms", []):
                critique_lines.append(f"• **{c.get('vulnerability')}:** {c.get('evidence_or_rationale')}")
            critiques_md = "\n\n".join(critique_lines)

            body = (
                f"### Verdict: [{v_color}]{verdict}[/{v_color}]\n\n"
                f"**Summary:** {res.get('summary')}\n\n"
                f"### Critical Vulnerabilities:\n{critiques_md}\n\n"
                f"### Hardware & VRAM Bottlenecks (RTX 4050 6GB):\n{res.get('hardware_and_vram_risk')}\n\n"
                f"### Minimal Disproving Experiment:\n{res.get('minimal_disproving_experiment')}"
            )
            console.print(Panel(Markdown(body), title=f"🎯 Devil's Advocate Review: [{v_color}]{verdict}[/{v_color}]", border_style=v_color, box=box.ROUNDED))

        elif cmd in ("/explain", "/deepdive"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[yellow]Usage: /explain <topic_or_architecture>[/yellow]")
                continue
            topic = parts[1].strip()
            last_query = topic

            with console.status("[cyan]Preparing comprehensive pedagogical deep-dive tutorial (math, tensor flow, PyTorch code)...[/cyan]"):
                res = pipeline.run(topic, top_k=config.RETRIEVAL_TOP_K, deep_dive=True)

            last_answer = res.get("answer", "")

            console.print()
            console.rule(f"[bold cyan]🎓 Deep-Dive Lecture: {topic}[/bold cyan]", style="cyan")
            console.print()
            console.print(Markdown(last_answer, code_theme="monokai"))
            console.print()
            console.rule(style="dim cyan")
            console.print()

        else:
            # Regular question -> Answer Pipeline
            last_query = user_input
            with console.status("[cyan]Retrieving papers, checking memory, and synthesizing answer...[/cyan]"):
                res = pipeline.run(user_input, top_k=config.RETRIEVAL_TOP_K)

            last_answer = res.get("answer", "")

            # Show Turkish warning if detected
            if res.get("warning"):
                console.print(f"[yellow]{res['warning']}[/yellow]\n")

            # Format Quotes
            quotes = res.get("quotes", [])
            if quotes:
                quote_lines = []
                for q in quotes:
                    mark = "[green]✓ [VERIFIED][/green]" if q.get("verified") else "[yellow]?[/yellow]"
                    quote_lines.append(f"{mark} *\"{q.get('text')}\"*\n  [dim]Source: {q.get('source')}[/dim]")
                quotes_str = "\n\n".join(quote_lines)
                ratio_color = "green" if res.get("verified_ratio", 0) >= 0.8 else "yellow"
                ratio_badge = f"[{ratio_color}]Attribution Score: {res.get('verified_ratio', 0)*100:.0f}%[/{ratio_color}]"
                console.print(Panel(quotes_str, title=f"Verbatim Grounding Quotes ({ratio_badge})", border_style="cyan", box=box.ROUNDED))

            # Render Synthesis with clean typography instead of cramped border
            console.print()
            console.rule("[bold green]🔬 Technical Synthesis & Architecture Analysis[/bold green]", style="green")
            console.print()
            console.print(Markdown(last_answer, code_theme="monokai"))
            console.print()
            console.rule(style="dim green")
            console.print()

            takeaways = res.get("key_takeaways", [])
            if takeaways:
                tk_list = "\n".join(f"• {t}" for t in takeaways)
                console.print(Panel(tk_list, title="Key Engineering Takeaways", border_style="blue", box=box.ROUNDED))


if __name__ == "__main__":
    run_repl()
