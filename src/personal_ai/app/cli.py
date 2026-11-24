"""Interactive Terminal CLI for Personal Engineering AI Assistant (Phases 1, 2, 3).

Run via:
    python -m personal_ai.app.cli
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from personal_ai.agents.specialists import AgentRegistry
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
    orchestrator = AgentOrchestrator(
        registry=registry,
        router=router,
        retriever=retriever,
        client=client,
    )

    rag_enabled = True
    active_agent_lock: Optional[str] = None  # None means automatic routing

    # Short aliases for quick switching
    agent_aliases = {
        "dl": "deep_learning",
        "cv": "computer_vision_uav",
        "uav": "computer_vision_uav",
        "math": "mathematics_engineering",
        "sw": "software_engineering",
        "software": "software_engineering",
    }

    console.print(Panel.fit(
        f"[bold cyan]Personal Engineering AI Assistant & Research Lab[/bold cyan] [dim](Phase 3: Multi-Agent)[/dim]\n"
        f"Backend: [green]{client.base_url}[/green] | Model: [yellow]{client.default_model}[/yellow]\n"
        f"Active Specialist: [bold magenta]{active_agent_lock or 'AUTO (TaskRouter Active)'}[/bold magenta]\n"
        f"RAG Knowledge Base: [bold {'green' if rag_enabled else 'red'}]{'ACTIVE' if rag_enabled else 'OFF'}[/bold {'green' if rag_enabled else 'red'}] ({retriever.default_collection})\n"
        f"Specialists: [dim]{', '.join(registry.list_agents())}[/dim]\n"
        f"Commands: [dim]/agent [dl|cv|math|sw|auto], /add <file>, /index <dir>, /rag [on|off], /info, /clear, /exit[/dim]",
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
                    f"Temperature: {client.default_temperature}\n"
                    f"Specialist Mode: {active_agent_lock or 'Automatic Routing'}\n"
                    f"Available Agents: {', '.join(registry.list_agents())}\n"
                    f"RAG Status: {'ACTIVE' if rag_enabled else 'DISABLED'} (Vault: {retriever.default_collection})",
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

            # Orchestrate response stream
            response_chunks = []
            is_clarification = False

            events = orchestrator.handle_query_stream(
                query=clean_input,
                conversation_history=conversation_history,
                force_agent_id=active_agent_lock,
                enable_rag=rag_enabled,
            )

            for event in events:
                if event.event_type == "routing":
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
                        console.print(f"[dim green]🔍 RAG Bağlamı: {', '.join(event.metadata['sources'])}[/dim green]")
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

        except KeyboardInterrupt:
            console.print("\n[dim]Çıkış yapılıyor...[/dim]")
            break
        except OllamaRuntimeError as e:
            console.print(f"\n[bold red][Hata][/bold red] {e}")
        except Exception as ex:
            console.print(f"\n[bold red][Beklenmeyen Hata][/bold red] {ex}")


if __name__ == "__main__":
    main()
