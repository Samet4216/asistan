const parentDoc = window.parent.document;

function setupQuoteButton() {
    if (parentDoc.getElementById("agy-quote-btn")) return;

    const btn = parentDoc.createElement("div");
    btn.id = "agy-quote-btn";
    btn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
        Alıntı Yap
    `;
    btn.style.position = "fixed";
    btn.style.display = "none";
    btn.style.zIndex = "999999";
    btn.style.background = "#0f172a";
    btn.style.color = "#38bdf8";
    btn.style.border = "1px solid #38bdf8";
    btn.style.borderRadius = "20px";
    btn.style.padding = "6px 14px";
    btn.style.fontSize = "13px";
    btn.style.fontWeight = "600";
    btn.style.fontFamily = "-apple-system, sans-serif";
    btn.style.cursor = "pointer";
    btn.style.boxShadow = "0 6px 20px rgba(0,0,0,0.6)";
    btn.style.transition = "transform 0.12s ease, background 0.12s ease";

    btn.onmouseenter = () => {
        btn.style.backgroundColor = "#1e293b";
        btn.style.transform = "scale(1.06)";
    };
    btn.onmouseleave = () => {
        btn.style.backgroundColor = "#0f172a";
        btn.style.transform = "scale(1.0)";
    };

    parentDoc.body.appendChild(btn);

    let activeSelectedText = "";

    function handleSelection() {
        const selection = parentDoc.getSelection();
        const selectedText = selection ? selection.toString().trim() : "";

        if (selectedText && selectedText.length > 5) {
            activeSelectedText = selectedText;
            try {
                const range = selection.getRangeAt(0);
                const rect = range.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0) {
                    btn.style.top = `${rect.top - 42}px`;
                    btn.style.left = `${rect.left + (rect.width / 2) - 45}px`;
                    btn.style.display = "flex";
                    btn.style.alignItems = "center";
                    return;
                }
            } catch (err) {}
        }
        btn.style.display = "none";
    }

    parentDoc.addEventListener("mouseup", () => setTimeout(handleSelection, 20));
    parentDoc.addEventListener("keyup", () => setTimeout(handleSelection, 30));
    parentDoc.addEventListener("selectionchange", () => {
        if (!parentDoc.getSelection().toString().trim()) {
            btn.style.display = "none";
        }
    });

    btn.addEventListener("mousedown", (e) => {
        e.preventDefault();
        e.stopPropagation();

        if (!activeSelectedText) return;

        const chatInput = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
        if (chatInput) {
            const quoteLines = activeSelectedText.split("\n").map(l => "> " + l).join("\n");
            const existingVal = chatInput.value ? chatInput.value.trim() + "\n\n" : "";
            const newVal = existingVal + quoteLines + "\n\n";

            const nativeSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, "value").set;
            nativeSetter.call(chatInput, newVal);
            chatInput.dispatchEvent(new Event("input", { bubbles: true }));

            chatInput.focus();
            chatInput.selectionStart = chatInput.selectionEnd = chatInput.value.length;
            chatInput.scrollIntoView({ behavior: "smooth" });
        }

        btn.style.display = "none";
    });
}

setTimeout(setupQuoteButton, 500);
