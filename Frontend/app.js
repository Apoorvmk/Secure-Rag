(() => {
  const roleOrder = ["intern", "employee", "manager", "board"];
  const roleNames = { intern: "Intern", employee: "Employee", manager: "Manager", board: "Board" };
  const roleReaders = {
    intern: "Interns",
    employee: "Interns and employees",
    manager: "Interns, employees, and managers",
    board: "All four roles",
  };
  const views = ["home", "login", "chat", "bunker"];
  const apiBaseUrl = (window.SECURERAG_API_BASE_URL || "").replace(/\/+$/, "");
  const state = { token: sessionStorage.getItem("northstar_token"), user: null, toastTimer: null };
  const $ = (selector) => document.querySelector(selector);

  function setView(view) {
    if (!views.includes(view)) view = "home";
    views.forEach((name) => {
      const section = document.getElementById(`${name}-view`);
      if (section) section.classList.toggle("hidden", name !== view);
    });
    if (state.user && (view === "chat" || view === "bunker")) {
      document.querySelectorAll(".nav-tab").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
      if (view === "bunker") loadDocuments();
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function authHeaders(extra = {}) {
    return { ...extra, ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}) };
  }

  async function api(path, options = {}) {
    const response = await fetch(`${apiBaseUrl}${path}`, { ...options, headers: authHeaders(options.headers || {}) });
    const type = response.headers.get("content-type") || "";
    const body = type.includes("application/json") ? await response.json() : null;
    if (response.status === 401 && state.token) signOut(false);
    if (!response.ok) throw new Error(body?.detail || `Request failed (${response.status})`);
    return body;
  }

  function updateNavigation(user) {
    state.user = user;
    $("#public-nav").classList.toggle("hidden", Boolean(user));
    $("#private-nav").classList.toggle("hidden", !user);
    if (user) {
      const name = user.name || "Team member";
      $("#user-label").textContent = `${name} · ${roleNames[user.role] || user.role}`;
      $("#bunker-role").textContent = `${roleNames[user.role] || user.role.toUpperCase()} ACCESS`;
      populateAccessSelect("#access-level", "#access-explainer");
      populateAccessSelect("#note-access", "#note-access-explainer");
    }
  }

  function signOut(showMessage = true) {
    sessionStorage.removeItem("northstar_token");
    state.token = null;
    updateNavigation(null);
    setView("home");
    if (showMessage) toast("You have logged out.");
  }

  function populateAccessSelect(selectSelector, explainerSelector) {
    const select = $(selectSelector);
    const rank = roleOrder.indexOf(state.user?.role);
    const canAssign = rank < 0 ? [] : roleOrder.slice(0, rank + 1);
    select.replaceChildren(...canAssign.map((role) => {
      const option = document.createElement("option");
      option.value = role;
      option.textContent = `${roleNames[role]} — readable by ${roleReaders[role].toLowerCase()}`;
      return option;
    }));
    select.value = canAssign.at(-1) || "";
    const describe = () => {
      $(explainerSelector).textContent = select.value
        ? `Readable by ${roleReaders[select.value]}. The server enforces this label on every request.`
        : "Your account does not have upload access.";
    };
    select.onchange = describe;
    describe();
  }

  function toast(message) {
    const element = $("#toast");
    element.textContent = message;
    element.classList.add("show");
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => element.classList.remove("show"), 3200);
  }

  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });
  $("#logout-button").addEventListener("click", () => signOut());

  $("#login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = $("#login-message");
    message.classList.remove("error");
    message.textContent = "Checking your company account…";
    try {
      const result = await api("/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: $("#email").value.trim(), password: $("#password").value }),
      });
      state.token = result.access_token;
      sessionStorage.setItem("northstar_token", state.token);
      updateNavigation({ name: result.name, role: result.role });
      $("#password").value = "";
      message.textContent = "";
      setView("chat");
      toast(`Welcome, ${result.name || "team member"}.`);
    } catch (error) {
      message.classList.add("error");
      message.textContent = error.message === "Invalid credentials" ? error.message : "We could not sign you in. Check your details or contact your administrator.";
    }
  });

 function appendMessage(text, who) {
    const wrap = document.createElement("div");

    if (who === "user") {
        wrap.className = "user-message";
        wrap.textContent = text;
    } else {
        wrap.className = "assistant-message";

        const meta = document.createElement("div");
        meta.className = "message-meta";
        meta.textContent = "SCOUT";

        const content = document.createElement("div");
        content.className = "assistant-content";

        // Restore escaped Markdown characters
        const markdown = String(text)
            .replace(/\\([|*_])/g, "$1")
            .replace(/\\<br\s*\/?>/gi, "\n");

        // Convert Markdown into safe HTML
        content.innerHTML = DOMPurify.sanitize(
            marked.parse(markdown, { gfm: true })
        );

        wrap.append(meta, content);
    }

    $("#messages").append(wrap);
    $("#messages").scrollTop = $("#messages").scrollHeight;

    return wrap;
}

  async function sendQuestion(question) {
    const trimmed = question.trim();
    if (!trimmed || !state.token) return;
    appendMessage(trimmed, "user");
    const pending = document.createElement("div");
    pending.className = "typing";
    pending.textContent = "Scout is looking through your approved documents…";
    $("#messages").append(pending);
    $("#messages").scrollTop = $("#messages").scrollHeight;
    $("#chat-input").value = "";
    $("#chat-input").disabled = true;
    $(".send-button").disabled = true;
    try {
      const result = await api("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed }),
      });
      pending.remove();
      appendAnswer(result);
    } catch (error) {
      pending.remove();
      appendMessage(error.message, "assistant");
    } finally {
      $("#chat-input").disabled = false;
      $(".send-button").disabled = false;
      $("#chat-input").focus();
    }
  }

  function appendAnswer(result) {
    const wrap = appendMessage(result.response || "I couldn't find a clear answer in the documents available to your role.", "assistant");
    const sources = Array.isArray(result.sources) ? result.sources.slice(0, 8) : [];
    if (!sources.length) return;
    const block = document.createElement("div");
    block.className = "answer-sources";
    const heading = document.createElement("span");
    heading.textContent = "SOURCES";
    block.append(heading);
    sources.forEach((source) => {
      const item = document.createElement("span");
      item.textContent = `${source.citation || ""} ${source.title || "Company document"} · ${source.source || ""}`;
      block.append(item);
    });
    wrap.append(block);
  }

  $("#chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    sendQuestion($("#chat-input").value);
  });
  $("#chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendQuestion(event.currentTarget.value);
    }
  });
  document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => sendQuestion(button.dataset.prompt));
  });

  function formatSize(size) {
    if (!Number.isFinite(size)) return "";
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(0)} KB`;
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  }

  function accessDescription(level) {
    return roleReaders[level] ? `READABLE BY ${roleReaders[level].toUpperCase()}` : "ACCESS RESTRICTED";
  }

  async function loadDocuments() {
    const list = $("#document-list");
    list.replaceChildren();
    const loading = document.createElement("div");
    loading.className = "empty-state";
    loading.textContent = "Loading the company library…";
    list.append(loading);
    try {
      const result = await api("/bunker");
      const documents = result.documents || [];
      $("#document-count").textContent = String(documents.length).padStart(2, "0");
      list.replaceChildren();
      if (!documents.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "The library is quiet for now. Add a document or write a quick notice to get started.";
        list.append(empty);
        return;
      }
      documents.forEach((document) => list.append(renderDocument(document)));
    } catch (error) {
      list.replaceChildren();
      const failure = document.createElement("div");
      failure.className = "empty-state";
      failure.textContent = error.message;
      list.append(failure);
    }
  }

  function renderDocument(item) {
    const row = document.createElement("article");
    row.className = "document-row";
    const icon = document.createElement("span");
    icon.className = "file-icon";
    const extension = (item.original_filename || "FILE").split(".").pop().toUpperCase();
    icon.textContent = extension.slice(0, 4);
    const info = document.createElement("div");
    info.className = "document-info";
    const title = document.createElement("strong");
    title.textContent = item.title || item.original_filename || "Company file";
    const detail = document.createElement("p");
    const kind = item.category === "image" ? "IMAGE" : item.category === "video" ? "VIDEO" : "TEXT DOCUMENT";
    const status = item.searchable ? "SEARCHABLE BY SCOUT" : "STORED IN BUNKER";
    detail.textContent = `${kind} · ${status}${item.size_bytes ? ` · ${formatSize(item.size_bytes)}` : ""}`;
    const access = document.createElement("p");
    access.className = "doc-access";
    access.textContent = `${(item.access_level || "").toUpperCase()} · ${accessDescription(item.access_level)}`;
    info.append(title, detail, access);
    const open = document.createElement("button");
    open.className = "document-action";
    open.type = "button";
    open.textContent = item.category === "image" || item.category === "video" ? "Preview" : "Open";
    open.addEventListener("click", () => openDocument(item));
    row.append(icon, info, open);
    return row;
  }

  async function openDocument(item) {
    try {
      const response = await fetch(`${apiBaseUrl}/bunker/files/${encodeURIComponent(item.document_id)}`, { headers: authHeaders() });
      if (response.status === 401) {
        signOut(false);
        throw new Error("Your session expired. Please log in again.");
      }
      if (!response.ok) throw new Error("This document is not available to your role.");
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      if (item.category === "image" || item.category === "video") {
        showMediaPreview(url, item);
      } else {
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = item.original_filename || "company-document";
        document.body.append(anchor);
        anchor.click();
        anchor.remove();
        setTimeout(() => URL.revokeObjectURL(url), 15000);
      }
    } catch (error) {
      toast(error.message);
    }
  }

  function showMediaPreview(url, item) {
    const overlay = document.createElement("div");
    overlay.className = "media-overlay";
    const panel = document.createElement("div");
    panel.className = "media-panel";
    const heading = document.createElement("div");
    heading.className = "media-heading";
    const label = document.createElement("strong");
    label.textContent = item.original_filename || "Bunker file";
    const close = document.createElement("button");
    close.className = "media-close";
    close.type = "button";
    close.textContent = "Close ×";
    close.addEventListener("click", () => { URL.revokeObjectURL(url); overlay.remove(); });
    heading.append(label, close);
    const media = document.createElement(item.category === "video" ? "video" : "img");
    media.src = url;
    if (item.category === "video") media.controls = true;
    else media.alt = item.title || "Company image";
    panel.append(heading, media);
    overlay.append(panel);
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) { URL.revokeObjectURL(url); overlay.remove(); }
    });
    document.body.append(overlay);
  }

  $("#file-input").addEventListener("change", (event) => {
    const files = [...event.currentTarget.files];
    $("#selected-files").textContent = files.map((file) => `${file.name} · ${formatSize(file.size)}`).join("\n");
  });

  $("#upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const files = [...$("#file-input").files];
    const message = $("#upload-message");
    message.classList.remove("error");
    if (!files.length) { message.textContent = "Choose one or more files first."; return; }
    const level = $("#access-level").value;
    message.textContent = `Uploading ${files.length} file${files.length === 1 ? "" : "s"}…`;
    try {
      const results = [];
      for (const file of files) {
        const form = new FormData();
        form.append("file", file);
        form.append("access_level", level);
        results.push(await api("/bunker/upload", { method: "POST", body: form }));
      }
      message.textContent = "";
      $("#file-input").value = "";
      $("#selected-files").textContent = "";
      toast(`${results.length} file${results.length === 1 ? "" : "s"} added to the Bunker.`);
      loadDocuments();
    } catch (error) {
      message.classList.add("error");
      message.textContent = error.message;
    }
  });

  $("#note-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = $("#note-message");
    message.classList.remove("error");
    message.textContent = "Publishing notice…";
    try {
      await api("/bunker/notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: $("#note-title").value, text: $("#note-text").value, access_level: $("#note-access").value }),
      });
      $("#note-title").value = "";
      $("#note-text").value = "";
      message.textContent = "";
      toast("Notice published and searchable by Scout.");
      loadDocuments();
    } catch (error) {
      message.classList.add("error");
      message.textContent = error.message;
    }
  });

  $("#refresh-documents").addEventListener("click", loadDocuments);

  async function restoreSession() {
    if (!state.token) return;
    try {
      const user = await api("/me");
      updateNavigation(user);
      setView("chat");
    } catch (_) {
      sessionStorage.removeItem("northstar_token");
      state.token = null;
      updateNavigation(null);
    }
  }

  restoreSession();
})();
