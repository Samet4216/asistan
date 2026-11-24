"""Interactive Terminal CLI for Personal Engineering AI Assistant (Phases 1-4).

Run via:
    python -m personal_ai.app.cli
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from personal_ai.agents.specialists import AgentRegistry
from personal_ai.memory.feedback_manager import FeedbackManager
from personal_ai.rag.retriever import DocumentRetriever
from personal_ai.routing.orchestrator import AgentOrchestrator
from personal_ai.routing.router import TaskRouter
from personal_ai.runtime.ollama_client import OllamaClient, OllamaRuntimeError


SYSTEM_ENGINEERING_PERSONA = """Sen kıdemli bir Derin Öğrenme, Bilgisayarlı Görü, Robotik ve Mekatronik Mühendisliği Araştırma Partnerisin.
Kullanıcının projelerinde (İHA/SİHA uçuş dinamiği, YOLO/SAHI hedef tespiti, Pekiştirmeli Öğrenme algoritmaları, kontrol teorisi ve C++/Python yazılım mimarisi) teknik mentorusun.

İLKELERİN:
1. DOĞAL VE AKICI ANLATIM: Asla 'Bölüm 1/N', zorlama yol haritası kalıpları veya şablon maddeler tekrarlama. Gerçek bir kıdemli mühendis meslektaşı gibi akıcı, berrak ve doğrudan konuş.
2. MATEMATİK & FORMÜLLER: Matematiksel ifadeleri LaTeX ($...$ veya $$...$$) formatında eksiksiz ve temiz yaz; değişkenleri açıkla.
3. KOD DİSİPLİNİ: Kullanıcı açıkça talep etmedikçe gereksiz kod blokları yığma; teorik, algoritmik ve mimari mantığı öne çıkar. Kod istendiğinde ise çalışan, test edilebilir ve endüstri standardında temiz kod üret.
4. BOZUK ASCII ŞEMALARI YASAKTIR: Devre veya mimarileri el yapımı bozuk ASCII karakterleriyle çizmeye çalışma; açıklamalarını kavramsal bloklar veya Markdown tablolarıyla yap.
5. SIFIR HALÜSİNASYON & KAYNAK DİSİPLİNİ: Bağlam dokümanları verildiğinde cevabında ilgili kaynakları ve sayfa numaralarını belirt (`[Kaynak: dosya, Sayfa: X]`). Bilmediğin veya spekülatif bir konuda tahmin uydurma; dürüstçe sınırlarını belirt.
"""


def main():
    console = Console()
    client = OllamaClient()
    registry = AgentRegistry()
    router = TaskRouter(registry=registry)
    retriever = DocumentRetriever()
    feedback_mgr = FeedbackManager()
    orchestrator = AgentOrchestrator(
        registry=registry,
        router=router,
        retriever=retriever,
        client=client,
    )

    rag_enabled = True
    active_agent_lock: Optional[str] = None  # None means automatic routing
    last_interaction: Optional[Dict[str, Any]] = None

    agent_aliases = {
        "dl": "deep_learning",
        "cv": "computer_vision_uav",
        "uav": "computer_vision_uav",
        "math": "mathematics_engineering",
        "sw": "software_engineering",
        "software": "software_engineering",
    }

    console.print(Panel.fit(
        f"[bold cyan]Personal Engineering AI Assistant & Research Lab[/bold cyan] [dim](Phase 4: Feedback & Memory)[/dim]\n"
        f"Backend: [green]{client.base_url}[/green] | Model: [yellow]{client.default_model}[/yellow]\n"
        f"Active Specialist: [bold magenta]{active_agent_lock or 'AUTO (TaskRouter Active)'}[/bold magenta]\n"
        f"RAG Knowledge Base: [bold {'green' if rag_enabled else 'red'}]{'ACTIVE' if rag_enabled else 'OFF'}[/bold {'green' if rag_enabled else 'red'}] ({retriever.default_collection})\n"
        f"Commands: [dim]/agent [id|auto], /fb [+1|-1] <not>, /fb list, /add <dosya>, /index <dizin>, /rag [on|off], /clear, /exit[/dim]",
        border_style="cyan"
    ))

    # Health check
    if not client.is_alive():
        console.print(
            f"[bold red][!] Ollama backend is unreachable at {client.base_url}![/bold red]\n"
            "Please start Ollama service using: [bold green]ollama serve[/bold green]"
        )
        sys.exit(1)

    conversation_history: List[Dict[str, str]] = []

    while True:
        try:
            mode_prefix = f"[{active_agent_lock}]" if active_agent_lock else "[auto]"
            user_input = Prompt.ask(f"\n[bold green]Sen {mode_prefix}[/bold green]")
            if not user_input.strip():
                continue

            clean_input = user_input.strip()

            # Commands
            if clean_input.lower() in ("/exit", "/quit", "q"):
                console.print("[dim]Oturum kapatıldı. İyi çalışmalar![/dim]")
                break
            elif clean_input.lower() == "/clear":
                conversation_history.clear()
                console.print("[dim]Sohbet geçmişi temizlendi.[/dim]")
                continue
            elif clean_input.lower() == "/info":
                models = client.list_models()
                console.print(Panel(
                    f"Model: {client.default_model}\n"
                    f"Available Models: {', '.join(models)}\n"
                    f"Context Window: {client.default_num_ctx} tokens\n"
                    f"Specialist Mode: {active_agent_lock or 'Automatic Routing'}\n"
                    f"Available Agents: {', '.join(registry.list_agents())}\n"
                    f"RAG Status: {'ACTIVE' if rag_enabled else 'DISABLED'} (Vault: {retriever.default_collection})\n"
                    f"Total Feedback Logged: {len(feedback_mgr.load_all_feedback())}",
                    title="System Status",
                    border_style="blue"
                ))
                continue
            elif clean_input.lower().startswith("/agent"):
                parts = clean_input.split(maxsplit=1)
                if len(parts) > 1:
                    target = parts[1].strip().lower()
                    if target in ("auto", "none", "reset"):
                        active_agent_lock = None
                        console.print("[green]Ajan yönlendirmesi OTOMATİK moda alındı (TaskRouter devrede).[/green]")
                    else:
                        full_id = agent_aliases.get(target, target)
                        if full_id in registry.list_agents():
                            active_agent_lock = full_id
                            agent_obj = registry.get_agent(full_id)
                            console.print(f"[bold magenta]Uzman kilitlendi:[/bold magenta] {agent_obj.name}")
                        else:
                            console.print(f"[bold red]Bilinmeyen ajan:[/bold red] '{target}'. Seçenekler: dl, cv, math, sw, auto")
                else:
                    current_name = registry.get_agent(active_agent_lock).name if active_agent_lock else "OTOMATİK"
                    console.print(f"Mevcut Uzman: [bold magenta]{current_name}[/bold magenta]")
                    console.print(f"Mevcut Tüm Ajanlar: {', '.join(registry.list_agents())}")
                continue
            elif clean_input.lower().startswith("/rag"):
                parts = clean_input.split(maxsplit=1)
                if len(parts) > 1 and parts[1].lower() in ("off", "disable", "false", "0"):
                    rag_enabled = False
                    console.print("[yellow]RAG doküman desteği devre dışı bırakıldı.[/yellow]")
                elif len(parts) > 1 and parts[1].lower() in ("on", "enable", "true", "1"):
                    rag_enabled = True
                    console.print("[green]RAG doküman desteği etkinleştirildi.[/green]")
                else:
                    console.print(f"RAG durumu: [bold]{'Açık' if rag_enabled else 'Kapalı'}[/bold]")
                continue
            elif clean_input.lower().startswith("/add "):
                filepath = clean_input[5:].strip().strip('"').strip("'")
                try:
                    num_chunks = retriever.ingest_file(filepath)
                    console.print(f"[bold green]✔[/bold green] '{Path(filepath).name}' başarıyla indekslendi ({num_chunks} parça).")
                except Exception as e:
                    console.print(f"[bold red][Hata][/bold red] Dosya indekslenemedi: {e}")
                continue
            elif clean_input.lower().startswith("/index "):
                dirpath = clean_input[7:].strip().strip('"').strip("'")
                try:
                    results = retriever.ingest_directory(dirpath)
                    total_chunks = sum(v for v in results.values() if v > 0)
                    console.print(f"[bold green]✔[/bold green] Dizin tarandı: {len(results)} dosya, {total_chunks} toplam parça indekslendi.")
                except Exception as e:
                    console.print(f"[bold red][Hata][/bold red] Dizin taranamadı: {e}")
                continue
            elif clean_input.lower().startswith("/fb"):
                parts = clean_input.split(maxsplit=2)
                subcmd = parts[1].lower() if len(parts) > 1 else "list"

                if subcmd == "list":
                    all_fb = feedback_mgr.load_all_feedback()[-5:]
                    if not all_fb:
                        console.print("[dim]Henüz kaydedilmiş geri bildirim bulunmuyor.[/dim]")
                    else:
                        table = Table(title="Son Geri Bildirim Kayıtları")
                        table.add_column("ID", style="cyan")
                        table.add_column("Ajan", style="magenta")
                        table.add_column("Puan", justify="center")
                        table.add_column("Onay", justify="center")
                        table.add_column("Soru / Düzeltme")
                        for fb in all_fb:
                            score_str = "[green]+1[/green]" if fb.rating > 0 else ("[red]-1[/red]" if fb.rating < 0 else "0")
                            verif_str = "[green]✔ Onaylı[/green]" if fb.is_verified_for_training else "[dim]Bekliyor[/dim]"
                            table.add_row(fb.feedback_id, fb.agent_id, score_str, verif_str, fb.correction or fb.query[:40])
                        console.print(table)
                    continue

                elif subcmd in ("+1", "1", "up"):
                    if not last_interaction:
                        console.print("[yellow]Puan verilecek önceki bir yanıt bulunamadı.[/yellow]")
                        continue
                    note = parts[2] if len(parts) > 2 else ""
                    fb = feedback_mgr.log_interaction(
                        query=last_interaction["query"],
                        response=last_interaction["response"],
                        agent_id=last_interaction["agent_id"],
                        model_version=client.default_model,
                        rating=1,
                        correction=note or None,
                        sources_used=last_interaction.get("sources", []),
                    )
                    console.print(f"[bold green]✔ Olumlu geri bildirim kaydedildi (ID: {fb.feedback_id}).[/bold green]")
                    continue

                elif subcmd in ("-1", "-1", "down"):
                    if not last_interaction:
                        console.print("[yellow]Düzeltme yapılacak önceki bir yanıt bulunamadı.[/yellow]")
                        continue
                    correction_text = parts[2] if len(parts) > 2 else Prompt.ask("[yellow]Lütfen doğru yanıtı veya eleştirinizi girin[/yellow]")
                    fb = feedback_mgr.log_interaction(
                        query=last_interaction["query"],
                        response=last_interaction["response"],
                        agent_id=last_interaction["agent_id"],
                        model_version=client.default_model,
                        rating=-1,
                        correction=correction_text,
                        sources_used=last_interaction.get("sources", []),
                    )
                    console.print(f"[bold green]✔ Düzeltme kaydı oluşturuldu (ID: {fb.feedback_id}).[/bold green] [dim]Kalite kontrolünden geçmeden eğitime eklenmez.[/dim]")
                    continue

                elif subcmd == "verify":
                    if len(parts) < 3:
                        console.print("[yellow]Kullanım: /fb verify <feedback_id> [train|val|test][/yellow]")
                        continue
                    args = parts[2].split()
                    fb_id = args[0]
                    split_choice = args[1] if len(args) > 1 else "train"
                    success = feedback_mgr.verify_entry(fb_id, approved=True, split=split_choice)
                    if success:
                        console.print(f"[bold green]✔ Geri bildirim {fb_id} doğrulandı ve '{split_choice}' kümesine atandı.[/bold green]")
                    else:
                        console.print(f"[bold red]Geri bildirim ID'si bulunamadı:[/bold red] {fb_id}")
                    continue

            # Orchestrate response stream
            response_chunks = []
            is_clarification = False
            last_agent_id = active_agent_lock or "unknown"
            last_sources: List[str] = []

            events = orchestrator.handle_query_stream(
                query=clean_input,
                conversation_history=conversation_history,
                force_agent_id=active_agent_lock,
                enable_rag=rag_enabled,
            )

            for event in events:
                if event.event_type == "routing":
                    last_agent_id = event.metadata.get("agent_id", last_agent_id)
                    console.print(f"[bold magenta]🤖 {event.content}[/bold magenta]")
                elif event.event_type == "clarification":
                    is_clarification = True
                    console.print(Panel(
                        f"[bold yellow]Sorunuzu daha kesin yanıtlayabilmek için netleştirme gerekiyor:[/bold yellow]\n\n"
                        f"{event.content}",
                        title="Netleştirme Talebi",
                        border_style="yellow"
                    ))
                    break
                elif event.event_type == "context":
                    if event.metadata.get("sources"):
                        last_sources = event.metadata["sources"]
                        console.print(f"[dim green]🔍 RAG Bağlamı: {', '.join(last_sources)}[/dim green]")
                    console.print("\n[bold cyan]Asistan:[/bold cyan]")
                elif event.event_type == "token":
                    sys.stdout.write(event.content)
                    sys.stdout.flush()
                    response_chunks.append(event.content)
                elif event.event_type == "error":
                    console.print(f"\n[bold red][Hata][/bold red] {event.content}")
                elif event.event_type == "done":
                    sys.stdout.write("\n")

            if not is_clarification and response_chunks:
                full_reply = "".join(response_chunks)
                conversation_history.append({"role": "user", "content": clean_input})
                conversation_history.append({"role": "assistant", "content": full_reply})
                last_interaction = {
                    "query": clean_input,
                    "response": full_reply,
                    "agent_id": last_agent_id,
                    "sources": last_sources,
                }

        except KeyboardInterrupt:
            console.print("\n[dim]Çıkış yapılıyor...[/dim]")
            break
        except OllamaRuntimeError as e:
            console.print(f"\n[bold red][Hata][/bold red] {e}")
        except Exception as ex:
            console.print(f"\n[bold red][Beklenmeyen Hata][/bold red] {ex}")


if __name__ == "__main__":
    main()
