import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Bot,
  ChevronDown,
  Copy,
  FileText,
  Menu,
  MoreHorizontal,
  Pencil,
  Plus,
  Search,
  Send,
  Settings,
  Sparkles,
  Trash2,
  X,
  LogOut,
} from "lucide-react";
import "./index.css";

const suggestions = [
  "What was Amazon's revenue in 2024?",
  "Compare Apple and Google revenue",
  "Summarize Meta's latest annual report",
  "What were Amazon's major expenses?",
];

function App() {
  // --- AUTH STATES ---
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem("retriva_token") || "");
  const [authMode, setAuthMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [isLoadingAuth, setIsLoadingAuth] = useState(false);

  // --- APP STATES ---
  const [chats, setChats] = useState([]);
  const [chatsReady, setChatsReady] = useState(false);
  const [chatError, setChatError] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const operationRef = useRef(null);
  const sessionRef = useRef(0);
  const [active, setActive] = useState(null);
  const [input, setInput] = useState("");
  const [search, setSearch] = useState("");
  const [mobile, setMobile] = useState(false);
  const [editing, setEditing] = useState(null);
  const [edit, setEdit] = useState("");
  const [menu, setMenu] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [temperature, setTemperature] = useState(0.1);
  const [showCitationPanel, setShowCitationPanel] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState(null);

  const ref = useRef(null);
  const fileInputRef = useRef(null);
  const chat = chats.find((c) => c.id === active);

  const filtered = useMemo(
    () => chats.filter((c) => c.title.toLowerCase().includes(search.toLowerCase())),
    [chats, search]
  );

  useEffect(() => {
    if (token) fetchUser();
  }, [token]);

  useEffect(() => {
    if (active) ref.current?.focus();
  }, [active]);

  const decodeChat = (saved) => ({
    ...saved,
    loading: false,
    messages: saved.messages.map((message) => ({
      ...message, sources: JSON.parse(message.sources || "[]"),
    })),
  });

  useEffect(() => {
    const controller = new AbortController();
    const session = ++sessionRef.current;
    operationRef.current = null;
    setChatBusy(false);
    setChatsReady(false);
    setChats([]);
    setActive(null);
    setChatError("");
    if (user && token) {
      const loadChats = async () => {
        try {
          const response = await fetch("http://localhost:8000/api/chats", {
            headers: { Authorization: `Bearer ${token}` },
            signal: controller.signal,
          });
          if (!response.ok) throw new Error("Unable to load chats. Please sign in again to retry.");
          const saved = (await response.json()).map(decodeChat);
          if (session !== sessionRef.current) return;
          setChats(saved);
          setActive(saved[0]?.id ?? null);
          setChatsReady(true);
        } catch (error) {
          if (!controller.signal.aborted && session === sessionRef.current) setChatError(error.message);
        }
      };
      loadChats();
    }
    return () => {
      controller.abort();
      sessionRef.current++;
    };
  }, [user, token]);

  const persistChat = async (snapshot) => {
    const response = await fetch("http://localhost:8000/api/chats", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({
        id: snapshot.id,
        title: snapshot.title,
        messages: snapshot.messages.map((message) => ({
          role: message.role, text: message.text, created_at: message.created_at,
          sources: JSON.stringify(message.sources || []),
        })),
      }),
    });
    if (!response.ok) throw new Error("Unable to save chat. Your latest changes may not be saved.");
    return decodeChat(await response.json());
  };

  // Serialize UI mutations and ignore results from a previous login session.
  const beginChatOperation = () => {
    if (!user || !token || !chatsReady || operationRef.current) return null;
    const operation = { session: sessionRef.current };
    operationRef.current = operation;
    setChatBusy(true);
    setChatError("");
    return operation;
  };
  const isCurrentOperation = (operation) => operation.session === sessionRef.current;
  const finishChatOperation = (operation) => {
    if (operationRef.current === operation) {
      operationRef.current = null;
      setChatBusy(false);
    }
  };

  const fetchUser = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/users/me", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
      } else {
        handleLogout();
      }
    } catch (error) {
      handleLogout();
    }
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    setAuthError("");
    setIsLoadingAuth(true);

    try {
      if (authMode === "login") {
        const formData = new FormData();
        formData.append("username", email);
        formData.append("password", password);

        const response = await fetch("http://localhost:8000/api/auth/login", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "Login failed");
        }

        const data = await response.json();
        localStorage.setItem("retriva_token", data.access_token);
        setToken(data.access_token);
      } else {
        const response = await fetch("http://localhost:8000/api/auth/signup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "Signup failed");
        }

        const loginFormData = new FormData();
        loginFormData.append("username", email);
        loginFormData.append("password", password);

        const loginResponse = await fetch("http://localhost:8000/api/auth/login", {
          method: "POST",
          body: loginFormData,
        });

        if (loginResponse.ok) {
          const data = await loginResponse.json();
          localStorage.setItem("retriva_token", data.access_token);
          setToken(data.access_token);
        }
      }
    } catch (error) {
      setAuthError(error.message);
    } finally {
      setIsLoadingAuth(false);
    }
  };

  const handleCitationClick = (citationNum, sources) => {
    const index = parseInt(citationNum) - 1;
    if (sources && sources[index]) {
      setSelectedCitation({
        number: citationNum,
        source: sources[index],
      });
      setShowCitationPanel(true);
    }
  };

  const formatChunkText = (text) => {
    if (!text) return <span className="text-[#606a66] italic">No content available.</span>;
    
    return text.split('\n').map((line, index) => {
      if (line.startsWith('## ')) {
        return <h3 key={index} className="text-base font-bold text-[#e8eceb] mt-4 mb-2 border-b border-[#222a26] pb-1">{line.replace('## ', '')}</h3>;
      }
      if (line.startsWith('# ')) {
        return <h2 key={index} className="text-lg font-bold text-[#65e6a5] mt-4 mb-2">{line.replace('# ', '')}</h2>;
      }
      if (line.trim() === '') return <br key={index} />;
      return <p key={index} className="mb-2 text-[#d2d9d5]">{line}</p>;
    });
  };

  const handleLogout = () => {
    sessionRef.current++;
    operationRef.current = null;
    setChatBusy(false);
    setChatsReady(false);
    setChatError("");
    setEditing(null);
    setInput("");
    localStorage.removeItem("retriva_token");
    setToken("");
    setUser(null);
    setChats([]);
    setActive(null);
    setShowCitationPanel(false);
  };

  const newChat = async () => {
    const operation = beginChatOperation();
    if (!operation) return;
    try {
      const saved = await persistChat({ title: "New conversation", messages: [] });
      if (!isCurrentOperation(operation)) return;
      setChats((p) => [saved, ...p]);
      setActive(saved.id);
      setInput("");
      setMobile(false);
    } catch (error) {
      if (isCurrentOperation(operation)) setChatError(error.message);
    } finally {
      finishChatOperation(operation);
    }
  };

  const del = async (id) => {
    const operation = beginChatOperation();
    if (!operation) return;
    try {
      const response = await fetch(`http://localhost:8000/api/chats/${id}`, {
        method: "DELETE", headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) throw new Error("Unable to delete chat. Please retry.");
      if (!isCurrentOperation(operation)) return;
      setChats((p) => p.filter((c) => c.id !== id));
      setActive((current) => current === id ? null : current);
      setMenu(null);
      setShowCitationPanel(false);
    } catch (error) {
      if (isCurrentOperation(operation)) setChatError(error.message);
    } finally {
      finishChatOperation(operation);
    }
  };

  const rename = (c) => {
    setEditing(c.id);
    setEdit(c.title);
    setMenu(null);
  };

  const save = async (id) => {
    const snapshot = chats.find((c) => c.id === id);
    if (!snapshot) return;
    const operation = beginChatOperation();
    if (!operation) return;
    try {
      const saved = await persistChat({ ...snapshot, title: edit.trim() || "New conversation" });
      if (!isCurrentOperation(operation)) return;
      setChats((p) => p.map((c) => c.id === id ? saved : c));
      setEditing(null);
    } catch (error) {
      if (isCurrentOperation(operation)) setChatError(error.message);
    } finally {
      finishChatOperation(operation);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || !token) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("http://localhost:8000/api/ingest", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      const data = await response.json();
      if (response.ok) alert(`Success! Added ${data.chunks} chunks to the knowledge base.`);
      else alert(`Error: ${data.detail}`);
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Failed to upload file.");
    } finally {
      setUploading(false);
      event.target.value = null;
    }
  };

  const send = async (raw = input) => {
    const text = raw.trim();
    if (!text) return;
    const operation = beginChatOperation();
    if (!operation) return;
    let snapshot = chat;
    try {
      if (!snapshot) {
        snapshot = await persistChat({ title: text.slice(0, 32), messages: [] });
        if (!isCurrentOperation(operation)) return;
        setChats((p) => [snapshot, ...p]);
        setActive(snapshot.id);
      }
      const history = snapshot.messages;
      snapshot = {
        ...snapshot,
        title: history.length ? snapshot.title : text.slice(0, 32) + (text.length > 32 ? "..." : ""),
        messages: [...history, { role: "user", text, sources: [] }],
      };
      // Persist the user turn before querying so a reload cannot lose it.
      snapshot = await persistChat(snapshot);
      if (!isCurrentOperation(operation)) return;
      const id = snapshot.id;
      setChats((p) => p.map((c) => c.id === id ? { ...snapshot, loading: true } : c));
      setInput("");
      let answer;
      try {
        const response = await fetch("http://localhost:8000/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
          body: JSON.stringify({ query: text, chat_history: history, temperature }),
        });
        if (!isCurrentOperation(operation)) return;
        if (!response.ok) {
          if (response.status === 401) {
            handleLogout();
            return;
          }
          throw new Error("Failed to get response");
        }
        const data = await response.json();
        answer = { role: "assistant", text: data.answer, sources: data.sources || [] };
      } catch (error) {
        answer = { role: "assistant", text: error.message || "Sorry, I couldn't connect to the backend.", sources: [] };
      }
      if (!isCurrentOperation(operation)) return;
      snapshot = { ...snapshot, messages: [...snapshot.messages, answer], loading: false };
      setChats((p) => p.map((c) => c.id === id ? snapshot : c));
      const saved = await persistChat(snapshot);
      if (isCurrentOperation(operation)) setChats((p) => p.map((c) => c.id === id ? saved : c));
    } catch (error) {
      if (isCurrentOperation(operation)) setChatError(error.message);
    } finally {
      finishChatOperation(operation);
    }
  };

  // --- AUTH UI ---
  if (!user) {
    return (
      <div className="h-screen w-full flex items-center justify-center bg-[#0b0e0d] text-[#e8eceb]">
        <div className="w-full max-w-md p-8 bg-[#101412] border border-[#202623] rounded-2xl shadow-2xl">
          <div className="flex items-center gap-3 mb-8 justify-center">
            <div className="grid h-12 w-12 place-items-center rounded-xl border border-[#31523f] bg-[#13251c]">
              <Sparkles size={24} className="text-[#65e6a5]" />
            </div>
            <div>
              <div className="text-2xl font-semibold">Retriva</div>
              <div className="text-xs uppercase tracking-wider text-[#65706b]">Finance AI</div>
            </div>
          </div>
          <h2 className="text-2xl font-bold mb-6 text-center">{authMode === "login" ? "Welcome Back" : "Create Account"}</h2>
          {authError && (
            <div className="mb-4 p-3 rounded-lg bg-red-900/20 border border-red-800 text-red-400 text-sm text-center">{authError}</div>
          )}
          <form onSubmit={handleAuth} className="space-y-4">
            <div>
              <label className="block text-sm mb-2 text-[#8a9490]">Email</label>
              <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="w-full px-4 py-3 bg-[#151a18] border border-[#27302c] rounded-lg outline-none focus:border-[#3d4b44] text-sm" required />
            </div>
            <div>
              <label className="block text-sm mb-2 text-[#8a9490]">Password</label>
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full px-4 py-3 bg-[#151a18] border border-[#27302c] rounded-lg outline-none focus:border-[#3d4b44] text-sm" required />
            </div>
            <button type="submit" disabled={isLoadingAuth} className="w-full py-3 bg-[#13251c] border border-[#31523f] text-[#65e6a5] rounded-lg font-medium hover:bg-[#1a2f26] disabled:opacity-50 flex items-center justify-center gap-2">
              {isLoadingAuth ? "Processing..." : authMode === "login" ? "Sign In" : "Sign Up"}
            </button>
          </form>
          <p className="mt-6 text-center text-sm text-[#65706b]">
            {authMode === "login" ? "Don't have an account? " : "Already have an account? "}
            <button onClick={() => { setAuthMode(authMode === "login" ? "signup" : "login"); setAuthError(""); }} className="text-[#65e6a5] hover:underline font-medium">
              {authMode === "login" ? "Sign Up" : "Sign In"}
            </button>
          </p>
        </div>
      </div>
    );
  }

  // --- MAIN APP UI WITH SPLIT VIEW ---
  return (
    <div className="h-screen w-full overflow-hidden bg-[#0b0e0d] text-[#e8eceb] flex">
      {/* Main Content Area - Shrinks when panel opens */}
      <div className={`flex-1 flex flex-col transition-all duration-300 ease-in-out ${showCitationPanel ? 'mr-[450px]' : ''}`}>
        <div className="flex h-full">
          {mobile && <div className="fixed inset-0 z-40 bg-black/60 lg:hidden" onClick={() => setMobile(false)} />}
          <aside className={`fixed inset-y-0 left-0 z-50 flex w-[285px] flex-col border-r border-[#202623] bg-[#101412] transition-transform duration-200 lg:static lg:translate-x-0 ${mobile ? "translate-x-0" : "-translate-x-full"}`}>
            <div className="flex h-[72px] items-center justify-between px-5">
              <div className="flex items-center gap-3">
                <div className="grid h-9 w-9 place-items-center rounded-xl border border-[#31523f] bg-[#13251c]"><Sparkles size={17} className="text-[#65e6a5]" /></div>
                <div>
                  <div className="text-[15px] font-semibold">Retriva</div>
                  <div className="text-[10px] uppercase tracking-[.18em] text-[#65706b]">Finance AI</div>
                </div>
              </div>
              <button className="rounded-lg p-2 text-[#6d7773] lg:hidden" onClick={() => setMobile(false)}><X size={18} /></button>
            </div>
            <div className="px-3">
              <button onClick={newChat} disabled={!chatsReady || chatBusy} className="flex w-full items-center gap-3 rounded-xl border border-[#27302c] bg-[#151a18] px-4 py-3 text-sm font-medium hover:border-[#3d4b44]">
                <Plus size={17} className="text-[#72e7ad]" /> New chat
              </button>
            </div>
            <div className="px-3 pt-5">
              <div className="relative">
                <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#68716e]" />
                <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search chats" className="w-full rounded-lg bg-[#151a18] py-2.5 pl-9 pr-3 text-xs outline-none placeholder:text-[#606a66] focus:border focus:border-[#2e3c35]" />
              </div>
            </div>
            <div className="mt-4 flex-1 overflow-y-auto px-2">
              <div className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-[.17em] text-[#59625f]">Conversations</div>
              <div className="space-y-1">
                {filtered.map((c) => (
                  <div key={c.id} className="group relative">
                    {editing === c.id ? (
                      <div className="flex rounded-lg bg-[#1b211e] p-1">
                        <input autoFocus value={edit} onChange={(e) => setEdit(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") save(c.id); if (e.key === "Escape") setEditing(null); }} className="min-w-0 flex-1 bg-transparent px-2 py-1.5 text-xs outline-none" />
                        <button onClick={() => save(c.id)} className="px-2 text-[11px] text-[#72e7ad]">Save</button>
                      </div>
                    ) : (
                      <button onClick={() => { setActive(c.id); setMobile(false); }} className={`flex w-full rounded-lg px-3 py-2.5 text-left text-xs ${active === c.id ? "bg-[#1b211e] text-[#edf2ef]" : "text-[#8a9490] hover:bg-[#171d1a]"}`}>
                        <span className="truncate pr-8">{c.title}</span>
                      </button>
                    )}
                    {editing !== c.id && (
                      <button onClick={(e) => { e.stopPropagation(); setMenu(menu === c.id ? null : c.id); }} className={`absolute right-1.5 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-[#737d79] opacity-0 group-hover:opacity-100 hover:bg-[#29302d] ${menu === c.id ? "opacity-100" : ""}`}>
                        <MoreHorizontal size={15} />
                      </button>
                    )}
                    {menu === c.id && (
                      <div className="absolute right-2 top-[42px] z-20 w-32 rounded-lg border border-[#2a322e] bg-[#171c1a] p-1 shadow-2xl">
                        <button onClick={() => rename(c)} disabled={chatBusy} className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-xs text-[#c8cfcc] hover:bg-[#232a27]"><Pencil size={13} /> Rename</button>
                        <button onClick={() => del(c.id)} disabled={chatBusy} className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-xs text-[#e48e8e] hover:bg-[#2b2020]"><Trash2 size={13} /> Delete</button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
            <div className="border-t border-[#202623] p-3">
              <button onClick={() => fileInputRef.current.click()} disabled={uploading} className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-xs text-[#7d8783] hover:bg-[#171d1a] disabled:opacity-50 mb-1">
                {uploading ? "Processing..." : "📄 Upload PDF"}
              </button>
              <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept=".pdf" className="hidden" />
              <button onClick={() => setShowSettings(true)} className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-xs text-[#7d8783] hover:bg-[#171d1a] mb-1">
                <Settings size={16} /> Settings
              </button>
              <div className="mt-2 flex items-center gap-3 px-3 py-2 border-t border-[#202623] pt-3">
                <div className="grid h-7 w-7 place-items-center rounded-full bg-[#26312b] text-[10px] font-bold text-[#65e6a5]">
                  {user.email ? user.email[0].toUpperCase() : "U"}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-[#cbd2cf] truncate">{user.email}</div>
                  <div className="text-[10px] text-[#606a66]">Authenticated</div>
                </div>
                <button onClick={handleLogout} className="text-[#606a66] hover:text-[#e48e8e] p-1" title="Logout">
                  <LogOut size={14} />
                </button>
              </div>
            </div>
          </aside>
          <main className="relative flex min-w-0 flex-1 flex-col">
            <header className="flex h-[64px] items-center border-b border-[#1c2320] px-4 sm:px-7">
              <button onClick={() => setMobile(true)} className="mr-3 rounded-lg p-2 text-[#8b9591] lg:hidden"><Menu size={19} /></button>
              <div className="text-sm font-medium text-[#bfc7c3]">{chat?.title || "New chat"}</div>
              <div className="ml-auto hidden items-center gap-2 rounded-full border border-[#28312d] bg-[#121715] px-3 py-1.5 text-[10px] text-[#6f7975] sm:flex">
                <span className="h-1.5 w-1.5 rounded-full bg-[#64d99d]" /> Knowledge base connected
              </div>
            </header>
            {chatError && <div role="alert" className="px-4 py-2 text-sm text-red-400">{chatError}</div>}
            {!chatsReady && !chatError && <div role="status" className="px-4 py-2 text-sm text-[#8a9490]">Loading chats…</div>}
            <div className="flex-1 overflow-y-auto">
              {!chat || chat.messages.length === 0 ? (
                <div className="mx-auto flex min-h-full max-w-3xl flex-col items-center justify-center px-5 pb-20">
                  <div className="mb-6 grid h-16 w-16 place-items-center rounded-2xl border border-[#294536] bg-[#13241b]"><Bot size={28} className="text-[#6de3a7]" /></div>
                  <h1 className="text-center text-3xl font-semibold tracking-[-.035em] sm:text-4xl">What can I find for you?</h1>
                  <p className="mt-3 max-w-md text-center text-sm leading-6 text-[#707a76]">Ask Retriva about your financial documents. Answers are grounded in your knowledge base.</p>
                  <div className="mt-9 grid w-full max-w-2xl grid-cols-1 gap-2 sm:grid-cols-2">
                    {suggestions.map((s, i) => (
                      <button key={i} onClick={() => send(s)} disabled={!chatsReady || chatBusy} className="group rounded-xl border border-[#252d29] bg-[#111614] p-4 text-left text-xs leading-5 text-[#9ba49f] transition hover:-translate-y-0.5 hover:border-[#385044] hover:text-[#d8dfdc]">
                        <div className="mb-2 grid h-7 w-7 place-items-center rounded-lg bg-[#19221e] text-[#6ee0a5]"><FileText size={14} /></div>
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="mx-auto max-w-3xl px-4 pb-36 pt-8 sm:px-7">
                  {chat.messages.map((m, i) => (
                    <div key={i} className={`mb-8 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                      {m.role === "assistant" ? (
                        <div className="flex max-w-[92%] gap-3 sm:max-w-[82%]">
                          <div className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg border border-[#294536] bg-[#13241b]"><Sparkles size={14} className="text-[#6de3a7]" /></div>
                          <div className="flex-1">
                            <div className="rounded-2xl rounded-tl-md border border-[#222a26] bg-[#111614] px-4 py-3.5 text-sm leading-7 text-[#d2d9d5]">
                              {m.text.split(/(\[\d+\])/g).map((part, idx) => {
                                if (/^\[\d+\]$/.test(part)) {
                                  const citationNum = part.replace(/[\[\]]/g, "");
                                  return (
                                    <button key={idx} onClick={() => handleCitationClick(citationNum, m.sources)} className="mx-0.5 inline-flex h-4 min-w-[16px] items-center justify-center rounded bg-[#13251c] px-1 text-[10px] font-bold text-[#65e6a5] hover:bg-[#1a2f26] hover:scale-110 transition-transform cursor-pointer" title={`View source ${citationNum}`}>
                                      {citationNum}
                                    </button>
                                  );
                                }
                                return <span key={idx}>{part}</span>;
                              })}
                            </div>
                            {m.sources && m.sources.length > 0 && (
                              <div className="mt-2 flex flex-wrap gap-2">
                                {m.sources.map((s, j) => (
                                  <button key={j} onClick={() => handleCitationClick(j + 1, m.sources)} className="flex items-center gap-1.5 rounded-full border border-[#29332e] bg-[#131916] px-2.5 py-1.5 text-[10px] text-[#7e8984] hover:border-[#65e6a5] hover:text-[#65e6a5] transition-colors cursor-pointer">
                                    <FileText size={11} className="text-[#63d99e]" />
                                    {s.company} · {s.year}
                                    <span className="text-[#56615c]">· {(s.relevance_score || s.score).toFixed(2)}</span>
                                  </button>
                                ))}
                              </div>
                            )}
                            <button className="mt-2 rounded-md p-1.5 text-[#59635f] hover:bg-[#171d1a]" title="Copy"><Copy size={13} /></button>
                          </div>
                        </div>
                      ) : (
                        <div className="max-w-[78%] rounded-2xl rounded-tr-md bg-[#dfe9e3] px-4 py-3 text-sm leading-6 text-[#17201c] sm:max-w-[68%]">{m.text}</div>
                      )}
                    </div>
                  ))}
                  {chat?.loading && (
                    <div className="mb-8 flex justify-start">
                      <div className="flex gap-3">
                        <div className="grid h-7 w-7 place-items-center rounded-lg border border-[#294536] bg-[#13241b]"><Sparkles size={14} className="text-[#6de3a7]" /></div>
                        <div className="rounded-2xl rounded-tl-md border border-[#222a26] bg-[#111614] px-4 py-3.5">
                          <div className="flex items-center gap-1.5">
                            <span className="dot-1 h-1.5 w-1.5 rounded-full bg-[#6de3a7]" />
                            <span className="dot-2 h-1.5 w-1.5 rounded-full bg-[#6de3a7]" />
                            <span className="dot-3 h-1.5 w-1.5 rounded-full bg-[#6de3a7]" />
                            <span className="ml-2 text-xs text-[#66716c]">Retriva is searching your knowledge base…</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-[#0b0e0d] via-[#0b0e0d]/95 to-transparent px-4 pb-5 pt-12 sm:px-7">
              <div className="pointer-events-auto mx-auto max-w-3xl">
                <form onSubmit={(e) => { e.preventDefault(); send(); }} className="relative rounded-2xl border border-[#303a35] bg-[#121715] p-2 shadow-2xl focus-within:border-[#46574e]">
                  <textarea ref={ref} rows={1} value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} placeholder="Ask Retriva about your financial documents..." className="min-h-[44px] w-full resize-none bg-transparent px-3 py-2.5 pr-14 text-sm outline-none placeholder:text-[#59635f]" disabled={!chatsReady || chatBusy} />
                  <button disabled={!input.trim() || !chatsReady || chatBusy} className="absolute bottom-2.5 right-2.5 grid h-9 w-9 place-items-center rounded-xl bg-[#dce9e1] text-[#17201b] disabled:opacity-30"><Send size={15} /></button>
                </form>
                <div className="mt-2 text-center text-[10px] text-[#4f5955]">Retriva can make mistakes. Verify important financial information.</div>
              </div>
            </div>
          </main>
        </div>
      </div>

      {/* Citation Panel - Slides in from right */}
      {showCitationPanel && selectedCitation && (
        <div className="fixed inset-y-0 right-0 w-[450px] bg-[#101412] border-l border-[#202623] shadow-2xl transform transition-transform duration-300 ease-in-out z-50 flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-[#202623]">
            <div className="flex items-center gap-3">
              <div className="grid h-8 w-8 place-items-center rounded-lg bg-[#13251c] border border-[#31523f]">
                <span className="text-base font-bold text-[#65e6a5]">{selectedCitation.number}</span>
              </div>
              <div>
                <h3 className="text-base font-semibold text-[#e8eceb]">Source Document</h3>
                <p className="text-xs text-[#65706b]">
                  {selectedCitation.source.company?.toUpperCase()} · {selectedCitation.source.year}
                </p>
              </div>
            </div>
            <button 
              onClick={() => setShowCitationPanel(false)}
              className="rounded-lg p-2 text-[#6d7773] hover:bg-[#171d1a] hover:text-white transition-colors"
            >
              <X size={18} />
            </button>
          </div>

          {/* Content - Scrollable */}
          <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5 custom-scrollbar">
            {/* Metadata */}
            <div className="rounded-xl border border-[#222a26] bg-[#0b0e0d] p-4">
              <h4 className="text-sm font-semibold text-[#cbd2cf] mb-3 flex items-center gap-2">
                <FileText size={14} className="text-[#63d99e]" />
                Document Metadata
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between py-1.5 border-b border-[#171d1a]">
                  <span className="text-[#606a66]">Company</span>
                  <span className="font-medium text-[#e8eceb]">{selectedCitation.source.company}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#171d1a]">
                  <span className="text-[#606a66]">Year</span>
                  <span className="font-medium text-[#e8eceb]">{selectedCitation.source.year}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-[#171d1a]">
                  <span className="text-[#606a66]">Document Type</span>
                  <span className="font-medium text-[#e8eceb]">{selectedCitation.source.type || "N/A"}</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-[#606a66]">Relevance Score</span>
                  <span className="font-medium text-[#65e6a5]">{(selectedCitation.source.relevance_score || selectedCitation.source.score || 0).toFixed(3)}</span>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="rounded-xl border border-[#222a26] bg-[#111614] p-4">
              <h4 className="text-sm font-semibold text-[#cbd2cf] mb-3 flex items-center gap-2">
                <FileText size={14} className="text-[#63d99e]" />
                Referenced Content
              </h4>
              <div className="rounded-lg bg-[#0b0e0d] p-4 border border-[#27302c]">
                <div className="text-sm leading-7 text-[#d2d9d5]">
                  {formatChunkText(selectedCitation.source.text)}
                </div>
              </div>
              <div className="mt-3 flex items-center gap-2 text-xs text-[#606a66]">
                <span className="inline-flex items-center gap-1 px-2 py-1 rounded bg-[#171d1a]">
                  <div className="h-1.5 w-1.5 rounded-full bg-[#65e6a5]"></div>
                  Source #{selectedCitation.number}
                </span>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="border-t border-[#202623] p-4 flex gap-3">
            <button
              onClick={() => setShowCitationPanel(false)}
              className="flex-1 rounded-lg border border-[#27302c] bg-[#151a18] px-4 py-2.5 text-sm text-[#8a9490] hover:border-[#3d4b44] hover:text-[#e8eceb] transition-colors"
            >
              Close
            </button>
            <button
              onClick={() => {
                navigator.clipboard.writeText(selectedCitation.source.text || "");
              }}
              className="flex-1 rounded-lg bg-[#13251c] border border-[#31523f] px-4 py-2.5 text-sm text-[#65e6a5] hover:bg-[#1a2f26] transition-colors flex items-center justify-center gap-2"
            >
              <Copy size={14} />
              Copy
            </button>
          </div>
        </div>
      )}

      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
          <div className="w-96 rounded-2xl border border-[#202623] bg-[#101412] p-6 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-[#e8eceb]">LLM Settings</h2>
              <button onClick={() => setShowSettings(false)} className="text-[#6d7773] hover:text-white"><X size={20} /></button>
            </div>
            <div className="space-y-6">
              <div>
                <div className="flex justify-between text-xs text-[#8a9490] mb-2">
                  <span>Temperature (Creativity)</span>
                  <span className="text-[#65e6a5]">{temperature.toFixed(2)}</span>
                </div>
                <input type="range" min="0.0" max="1.0" step="0.05" value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))} className="w-full accent-[#65e6a5]" />
                <p className="mt-2 text-[10px] text-[#59625f]">Low (0.1) = Strict & Accurate (Best for Finance). High (0.8) = Creative.</p>
              </div>
            </div>
            <button onClick={() => setShowSettings(false)} className="mt-8 w-full rounded-lg bg-[#13251c] py-2.5 text-sm font-medium text-[#65e6a5] hover:bg-[#1a2f26]">Save Settings</button>
          </div>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);