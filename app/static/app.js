/* DocuAsk frontend — plain ES2019, no framework, no build step.
 *
 * Security note: every string that comes back from the API (document names,
 * answers, snippets) is written with textContent / createElement only. There
 * is no innerHTML anywhere in this file, so a PDF whose filename or text
 * contains markup cannot inject into the page.
 */

"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  documents: [],
  sending: false,
  uploading: false,
  conversation: [],
};

/* --- one place that speaks the API envelope ---------------------------- */

class ApiError extends Error {
  constructor(message, code, status) {
    super(message);
    this.name = "ApiError";
    this.code = code || "unknown";
    this.status = status || 0;
  }
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (networkError) {
    throw new ApiError("Could not reach the server. Is it still running?", "network", 0);
  }

  let body = null;
  try {
    body = await response.json();
  } catch (parseError) {
    throw new ApiError(`The server returned something that was not JSON (HTTP ${response.status}).`,
      "bad_response", response.status);
  }

  if (!body.success) {
    const error = body.error || {};
    throw new ApiError(error.message || `Request failed (HTTP ${response.status}).`,
      error.code, response.status);
  }
  return body.data;
}

/* --- small DOM helpers (never innerHTML) ------------------------------ */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function setStatus(node, text, kind) {
  node.textContent = text || "";
  if (kind) node.setAttribute("data-state", kind);
  else node.removeAttribute("data-state");
}

function showError(node, error) {
  const where = error.status ? ` (HTTP ${error.status})` : "";
  const code = error.code && error.code !== "unknown" ? ` [${error.code}]` : "";
  node.textContent = `${error.message}${where}${code}`;
  node.hidden = false;
}

function hideError(node) {
  node.hidden = true;
  node.textContent = "";
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

/* --- health ------------------------------------------------------------ */

async function loadHealth() {
  const badge = $("health");
  try {
    const data = await request("/health");
    badge.setAttribute("data-state", data.status === "ok" ? "ok" : "degraded");
    if (data.status === "ok") {
      badge.textContent = "All services healthy";
    } else {
      const llm = data.llm || {};
      const reason = !llm.reachable
        ? "Ollama not reachable — chat will not work"
        : !llm.model_available
          ? `Model '${llm.model}' not pulled`
          : "A dependency is degraded";
      badge.textContent = reason;
      badge.title = llm.detail || reason;
    }
  } catch (error) {
    badge.setAttribute("data-state", "error");
    badge.textContent = "Server unreachable";
    badge.title = error.message;
  }
}

/* --- document library -------------------------------------------------- */

function renderDocuments() {
  const list = $("doc-list");
  const empty = $("library-empty");
  const counter = $("doc-count");
  const scope = $("scope-select");

  // Rebuilding the options must not silently reset the user's scope choice.
  const previousScope = scope.value;

  clear(list);
  clear(scope);

  const all = document.createElement("option");
  all.value = "";
  all.textContent = "All documents";
  scope.appendChild(all);

  const count = state.documents.length;
  counter.textContent = count ? `${count} indexed` : "";
  empty.hidden = count > 0;
  list.hidden = count === 0;

  for (const doc of state.documents) {
    const item = el("li", "doc");

    const info = el("div", "doc-body");
    info.appendChild(el("p", "doc-name", doc.filename));
    info.appendChild(el("p", "doc-meta",
      `${doc.page_count} page${doc.page_count === 1 ? "" : "s"} · ` +
      `${doc.chunk_count} chunks · ${formatBytes(doc.size_bytes)}`));

    const remove = el("button", "doc-delete", "Delete");
    remove.type = "button";
    remove.setAttribute("aria-label", `Delete ${doc.filename}`);
    remove.addEventListener("click", () => deleteDocument(doc, remove));

    item.appendChild(info);
    item.appendChild(remove);
    list.appendChild(item);

    const option = document.createElement("option");
    option.value = doc.id;
    option.textContent = doc.filename;
    scope.appendChild(option);
  }

  // Keep the previous choice while its document is still listed (an upload or
  // a deletion rebuilds this list); otherwise fall back to "All documents".
  for (const option of scope.options) {
    if (option.value === previousScope) {
      scope.value = previousScope;
      break;
    }
  }
}

async function loadDocuments() {
  try {
    const data = await request("/documents?limit=100");
    state.documents = data.documents || [];
    renderDocuments();
  } catch (error) {
    const status = $("upload-status");
    setStatus(status, error.message, "error");
  }
}

async function deleteDocument(doc, button) {
  if (!window.confirm(`Delete "${doc.filename}"? Its file, vectors and metadata are removed.`)) {
    return;
  }
  button.disabled = true;
  setStatus($("upload-status"), `Deleting ${doc.filename}…`, "busy");
  try {
    await request(`/documents/${encodeURIComponent(doc.id)}`, { method: "DELETE" });
    state.documents = state.documents.filter((item) => item.id !== doc.id);
    renderDocuments();
    setStatus($("upload-status"), `Deleted ${doc.filename} (${doc.chunk_count} chunks removed).`, "ok");
  } catch (error) {
    button.disabled = false;
    setStatus($("upload-status"), error.message, "error");
  }
}

/* --- upload ------------------------------------------------------------ */

function selectedFile() {
  const input = $("file-input");
  return input.files && input.files.length ? input.files[0] : null;
}

function refreshUploadButton() {
  $("upload-submit").disabled = !selectedFile() || state.uploading;
}

async function uploadFile(file) {
  const status = $("upload-status");
  state.uploading = true;
  refreshUploadButton();
  setStatus(status, `Indexing ${file.name}… (extraction, chunking and embedding run now)`, "busy");

  const form = new FormData();
  form.append("file", file, file.name);

  try {
    const data = await request("/documents/upload", { method: "POST", body: form });
    const doc = data.document;
    setStatus(status,
      `Indexed ${doc.filename}: ${doc.page_count} pages → ${doc.chunk_count} chunks ` +
      `in ${data.timings.total_seconds.toFixed(2)} s.`, "ok");
    state.documents.unshift(doc);
    renderDocuments();
    $("file-input").value = "";
    $("dropzone-name").textContent = "";
    hideError($("chat-error"));
  } catch (error) {
    setStatus(status, error.message, "error");
  } finally {
    state.uploading = false;
    refreshUploadButton();
  }
}

/* --- chat -------------------------------------------------------------- */

function scrollMessagesToBottom() {
  const box = $("messages");
  box.scrollTop = box.scrollHeight;
}

function addUserMessage(text) {
  $("chat-empty").hidden = true;
  const message = el("div", "message message-user");
  message.appendChild(el("p", "message-role", "You"));
  message.appendChild(el("p", "message-text", text));
  $("messages").appendChild(message);
  scrollMessagesToBottom();
  return message;
}

function addLoadingMessage() {
  const message = el("div", "message message-assistant is-loading");
  message.appendChild(el("p", "message-role", "DocuAsk"));
  const loading = el("p", "loading");
  loading.appendChild(el("span", "dot"));
  loading.appendChild(el("span", "dot"));
  loading.appendChild(el("span", "dot"));
  loading.appendChild(el("span", "sr-only", "Searching your documents and thinking"));
  message.appendChild(loading);
  $("messages").appendChild(message);
  scrollMessagesToBottom();
  return message;
}

function addAssistantMessage(data) {
  const message = el("div", "message message-assistant");
  message.setAttribute("data-grounded", String(Boolean(data.grounded)));
  message.appendChild(el("p", "message-role", data.grounded ? "DocuAsk" : "No match found"));
  message.appendChild(el("p", "message-text", data.answer));

  if (data.sources && data.sources.length) {
    const list = el("ul", "sources");
    list.setAttribute("aria-label", "Sources for this answer");
    for (const source of data.sources) {
      const item = el("li", "source");

      const head = el("div", "source-head");
      head.appendChild(el("span", "source-file", source.filename));
      head.appendChild(el("span", "source-page", `page ${source.page}`));
      head.appendChild(el("span", "source-score", `similarity ${source.similarity.toFixed(3)}`));
      item.appendChild(head);
      item.appendChild(el("p", "source-snippet", source.snippet));

      list.appendChild(item);
    }
    message.appendChild(list);
  }

  const timings = data.timings || {};
  const meta = [];
  if (data.grounded) {
    meta.push(`${data.context_chunks} of ${data.retrieved} chunks sent to the model`);
    if (data.context_truncated) meta.push("context was trimmed to fit the budget");
  } else {
    // An unexplained refusal reads like a bug. Say how close the library came.
    const closest = data.closest_similarity;
    const floor = data.min_similarity;
    if (typeof closest === "number" && typeof floor === "number") {
      meta.push(`closest match scored ${closest.toFixed(3)}, under the ${floor.toFixed(2)} floor`);
    } else {
      meta.push("the search returned no candidates");
    }
  }
  meta.push(`retrieval ${(timings.retrieval_seconds || 0).toFixed(3)} s`);
  meta.push(`answer in ${(timings.total_seconds || 0).toFixed(2)} s`);
  message.appendChild(el("p", "message-meta", meta.join(" · ")));

  $("messages").appendChild(message);
  scrollMessagesToBottom();
}

async function sendQuestion(question) {
  const errorBox = $("chat-error");
  hideError(errorBox);
  $("clear-chat").disabled = false;

  addUserMessage(question);
  const loading = addLoadingMessage();

  state.sending = true;
  $("send").disabled = true;
  $("question-input").disabled = true;

  const scope = $("scope-select").value;
  const payload = { question };
  if (scope) payload.document_ids = [scope];

  try {
    const data = await request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    loading.remove();
    addAssistantMessage(data);
    state.conversation.push({ question, answer: data.answer });
  } catch (error) {
    // The refusal case (grounded: false) arrives as data, not an error, so
    // anything caught here is a real failure: 409 no documents, 503 Ollama
    // down, 422 validation, network loss.
    loading.remove();
    // 409 no_documents, 503 llm_unavailable, 422 validation: the server's
    // message is already specific and actionable, so it is shown verbatim.
    // Nothing about the question is echoed back to the page.
    showError(errorBox, error);
  } finally {
    state.sending = false;
    $("send").disabled = false;
    $("question-input").disabled = false;
    $("question-input").focus();
    scrollMessagesToBottom();
  }
}

function clearConversation() {
  const box = $("messages");
  clear(box);
  const empty = el("p", "empty", "Answers cite a document, a page and a similarity score.");
  empty.id = "chat-empty";
  box.appendChild(empty);
  state.conversation = [];
  $("clear-chat").disabled = true;
  hideError($("chat-error"));
  $("question-input").focus();
}

/* --- wiring ------------------------------------------------------------ */

function wireUpload() {
  const input = $("file-input");
  const dropzone = $("dropzone");

  input.addEventListener("change", () => {
    const file = selectedFile();
    $("dropzone-name").textContent = file ? file.name : "";
    refreshUploadButton();
  });

  // Drag and drop is an enhancement; the file input remains the real control.
  ["dragenter", "dragover"].forEach((type) => {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-over");
    });
  });
  ["dragleave", "drop"].forEach((type) => {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-over");
    });
  });
  dropzone.addEventListener("drop", (event) => {
    const dropped = event.dataTransfer && event.dataTransfer.files;
    if (!dropped || !dropped.length) return;
    input.files = dropped;
    $("dropzone-name").textContent = dropped[0].name;
    refreshUploadButton();
  });

  $("upload-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const file = selectedFile();
    if (file) uploadFile(file);
  });
}

function wireChat() {
  const input = $("question-input");

  $("chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question || state.sending) return;
    input.value = "";
    sendQuestion(question);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("chat-form").requestSubmit();
    }
  });

  $("clear-chat").addEventListener("click", clearConversation);
}

document.addEventListener("DOMContentLoaded", () => {
  wireUpload();
  wireChat();
  loadHealth();
  loadDocuments();
  // Health is checked again whenever the tab comes back, so a restarted
  // Ollama is noticed without a manual refresh.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) loadHealth();
  });
});
