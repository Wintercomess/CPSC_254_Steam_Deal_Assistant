// Steam Deal Assistant frontend.
// Keeps multi-turn state client-side so follow-ups reuse extracted preferences.

const chatEl = document.getElementById("chat");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");

const state = {
  history: [],         // [{role, content}]
  preferences: null,   // last UserPreferences returned by the server
};

function addUserBubble(text) {
  const div = document.createElement("div");
  div.className = "bubble user";
  div.textContent = text;
  chatEl.appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
}

function addAssistantBubble(payload) {
  const wrap = document.createElement("div");
  wrap.className = "bubble assistant";

  if (payload.summary) {
    const s = document.createElement("div");
    s.className = "summary";
    s.textContent = payload.summary;
    wrap.appendChild(s);
  }

  if (!payload.recommendations || payload.recommendations.length === 0) {
    const empty = document.createElement("div");
    empty.textContent = "No matching deals in the catalog right now.";
    wrap.appendChild(empty);
  }

  for (const rec of payload.recommendations || []) {
    const card = document.createElement("div");
    card.className = "rec";

    const row = document.createElement("div");
    row.className = "row";

    const title = document.createElement("h4");
    title.textContent = rec.game_name;

    const verdict = document.createElement("span");
    verdict.className = `badge ${rec.verdict}`;
    verdict.textContent = rec.verdict;

    const price = document.createElement("span");
    price.className = "price";
    price.textContent = `$${Number(rec.current_price).toFixed(2)}`;

    const low = document.createElement("span");
    low.className = "badge";
    low.textContent = `low: $${Number(rec.historical_low).toFixed(2)}`;

    row.append(title, verdict, price, low);

    const reason = document.createElement("div");
    reason.className = "reason";
    reason.textContent = rec.reason;

    card.append(row, reason);
    wrap.appendChild(card);
  }

  if (payload.verification && payload.verification.retried) {
    const note = document.createElement("div");
    note.className = "verify-note";
    note.textContent = "ⓘ Output failed first-pass verification and was auto-corrected.";
    wrap.appendChild(note);
  }

  chatEl.appendChild(wrap);
  wrap.scrollIntoView({ behavior: "smooth", block: "end" });
}

async function send(message) {
  errorEl.hidden = true;
  errorEl.textContent = "";
  statusEl.textContent = "Thinking…";
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history: state.history,
        preferences: state.preferences,
      }),
    });

    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`Server error (${res.status}): ${detail}`);
    }

    const data = await res.json();
    state.preferences = data.preferences;
    state.history.push({ role: "user", content: message });

    // Assistant turn for the next round: a compact textual summary so the LLM
    // has context for follow-ups without re-sending all candidate JSON.
    const assistantText = [
      data.summary,
      ...(data.recommendations || []).map(
        (r) => `- ${r.game_name} (${r.verdict}, $${r.current_price}): ${r.reason}`
      ),
    ].join("\n");
    state.history.push({ role: "assistant", content: assistantText });

    addAssistantBubble(data);
  } catch (err) {
    errorEl.hidden = false;
    errorEl.textContent = err.message || "Something went wrong.";
  } finally {
    statusEl.textContent = "";
    sendBtn.disabled = false;
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const msg = inputEl.value.trim();
  if (!msg) return;
  addUserBubble(msg);
  inputEl.value = "";
  send(msg);
});

// Allow Enter to submit; Shift+Enter for newline.
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});
