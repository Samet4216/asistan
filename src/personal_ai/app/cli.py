"""Interactive Terminal CLI for Personal Engineering AI Assistant (Phase 1).

Run via:
    python -m personal_ai.app.cli
"""

import sys
from typing import Dict, List
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from personal_ai.runtime.ollama_client import OllamaClient, OllamaRuntimeError


SYSTEM_ENGINEERING_PERSONA = """Sen kıdemli bir Derin Öğrenme, Bilgisayarlı Görü, Robotik ve Mekatronik Mühendisliği Araştırma Partnerisin.
Kullanıcının projelerinde (İHA/SİHA uçuş dinamiği, YOLO/SAHI hedef tespiti, Pekiştirmeli Öğrenme algoritmaları, kontrol teorisi ve C++/Python yazılım mimarisi) teknik mentorusun.

İLKELERİN:
1. DOĞAL VE AKICI ANLATIM: Asla 'Bölüm 1/N', zorlama yol haritası kalıpları veya şablon maddeler tekrarlama. Gerçek bir kıdemli mühendis meslektaşı gibi akıcı, berrak ve doğrudan konuş.
2. MATEMATİK & FORMÜLLER: Matematiksel ifadeleri LaTeX ($...$ veya $$...$$) formatında eksiksiz ve temiz yaz; değişkenleri açıkla.
3. KOD DİSİPLİNİ: Kullanıcı açıkça talep etmedikçe gereksiz kod blokları yığma; teorik, algoritmik ve mimari mantığı öne çıkar. Kod istendiğinde ise çalışan, test edilebilir ve endüstri standardında temiz kod üret.
4. BOZUK ASCII ŞEMALARI YASAKTIR: Devre veya mimarileri el yapımı bozuk ASCII karakterleriyle çizmeye çalışma; açıklamalarını kavramsal bloklar veya Markdown tablolarıyla yap.
5. SIFIR HALÜSİNASYON: Bilmediğin veya spekülatif bir konuda tahmin uydurma; dürüstçe sınırlarını belirt.
"""


def main():
    console = Console()
    client = OllamaClient()

    console.print(Panel.fit(
        f"[bold cyan]Personal Engineering AI Assistant[/bold cyan] [dim](Phase 1 CLI)[/dim]\n"
        f"Backend: [green]{client.base_url}[/green] | Model: [yellow]{client.default_model}[/yellow] | Temp: [magenta]{client.default_temperature}[/magenta]\n"
        f"Commands: [dim]/exit, /clear, /info[/dim]",
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
                    f"Temperature: {client.default_temperature}",
                    title="System Status",
                    border_style="blue"
                ))
                continue

            conversation_history.append({"role": "user", "content": clean_input})

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
