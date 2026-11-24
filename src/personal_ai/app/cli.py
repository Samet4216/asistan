"""Interactive Terminal CLI for Personal Engineering AI Assistant (Phase 1 & Phase 2).

Run via:
    python -m personal_ai.app.cli
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from personal_ai.rag.retriever import DocumentRetriever
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
    retriever = DocumentRetriever()
    rag_enabled = True

    console.print(Panel.fit(
        f"[bold cyan]Personal Engineering AI Assistant & Research Lab[/bold cyan]\n"
        f"Backend: [green]{client.base_url}[/green] | Model: [yellow]{client.default_model}[/yellow] | Temp: [magenta]{client.default_temperature}[/magenta]\n"
        f"RAG Knowledge Base: [bold {'green' if rag_enabled else 'red'}]{'ACTIVE' if rag_enabled else 'OFF'}[/bold {'green' if rag_enabled else 'red'}] ({retriever.default_collection})\n"
        f"Commands: [dim]/add <path>, /index <dir>, /rag [on|off], /info, /clear, /exit[/dim]",
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
            user_input = Prompt.ask("\n[bold green]Sen[/bold green]")
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
                    f"RAG Status: {'ACTIVE' if rag_enabled else 'DISABLED'} (Vault: {retriever.default_collection})\n"
                    f"Chroma Directory: {retriever.persist_directory}",
                    title="System Status",
                    border_style="blue"
                ))
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

            # Query retrieval
            augmented_query = clean_input
            if rag_enabled:
                retrieval = retriever.retrieve(clean_input)
                if retrieval.has_context:
                    console.print(f"[dim green]🔍 RAG Bağlamı Bulundu ({len(retrieval.chunks)} kaynak): {', '.join(retrieval.sources_summary)}[/dim green]")
                    augmented_query = retriever.build_rag_prompt(clean_input, retrieval)
                else:
                    console.print("[dim yellow]ℹ Yüklenen dokümanlarda doğrudan eşleşme bulunamadı; genel mühendislik bilgisiyle yanıtlanıyor.[/dim yellow]")

            conversation_history.append({"role": "user", "content": augmented_query})

            console.print("\n[bold cyan]Asistan:[/bold cyan]")
            response_chunks = []
            
            # Streaming live tokens
            for token in client.chat_stream(
                messages=conversation_history,
                system_prompt=SYSTEM_ENGINEERING_PERSONA
            ):
                sys.stdout.write(token)
                sys.stdout.flush()
                response_chunks.append(token)
            
            full_response = "".join(response_chunks)
            sys.stdout.write("\n")
            # In history, preserve the user's natural query to not pollute next-turn context
            conversation_history[-1]["content"] = clean_input
            conversation_history.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            console.print("\n[dim]Çıkış yapılıyor...[/dim]")
            break
        except OllamaRuntimeError as e:
            console.print(f"\n[bold red][Hata][/bold red] {e}")
        except Exception as ex:
            console.print(f"\n[bold red][Beklenmeyen Hata][/bold red] {ex}")


if __name__ == "__main__":
    main()
