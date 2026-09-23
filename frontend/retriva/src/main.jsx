import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Bot, ChevronDown, Copy, FileText, Menu, MoreHorizontal,
  Pencil, Plus, Search, Send, Settings, Sparkles, Square, Trash2, X, LogOut, User,
} from "lucide-react";
import "./index.css";

const suggestions = [
  "What was Amazon's revenue in 2024?",
  "Compare Apple and Google revenue",
  "Summarize Meta's latest annual report",
  "What were Amazon's major expenses?",
];

// API base URL. Empty string => same-origin (nginx proxies /api to the backend),
// which is what we use in containers. For a separately-hosted frontend (S3/CDN)
// set VITE_API_URL at build time, e.g. https://api.example.com
const API = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");

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
  const [active, setActive] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [search, setSearch] = useState("");
  const [mobile, setMobile] = useState(false);
  const [editing, setEditing] = useState(null);
  const [edit, setEdit] = useState("");
  const [menu, setMenu] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [temperature, setTemperature] = useState(0.1);
  const [bio, setBio] = useState("");
  const [bioDraft, setBioDraft] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [profileSaved, setProfileSaved] = useState(false);
  const [showCitationPanel, setShowCitationPanel] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const ref = useRef(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const abortRef = useRef(null);

  const filtered = useMemo(
    () => chats.filter((c) => c.title.toLowerCase().includes(search.toLowerCase())),
    [chats, search]
  );

  // --- AUTO SCROLL ---
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // --- FETCH USER ON TOKEN ---
  useEffect(() => {
    if (token) fetchUser();
  }, [token]);

  // --- FETCH MESSAGES WHEN ACTIVE CHAT CHANGES ---
  useEffect(() => {
    if (active && token) {
      fetchMessages(active);
    } else {
      setMessages([]);
    }
  }, [active, token]);

  // --- FOCUS INPUT ---
  useEffect(() => {
    if (active) ref.current?.focus();
  }, [active]);

  // ===================== API CALLS =====================

  const authHeaders = () => ({ Authorization: `Bearer ${token}` });

  const fetchUser = async () => {
    try {
      const res = await fetch(`${API}/api/users/me`, { headers: authHeaders() });
      if (res.ok) {
        setUser(await res.json());
        fetchChats();
        fetchProfile();
      } else {
        handleLogout();
      }
    } catch {
      handleLogout();
    }
  };

  const fetchProfile = async () => {
    try {
      const res = await fetch(`${API}/api/profile`, { headers: authHeaders() });
      if (res.ok) {
        const data = await res.json();
        setBio(data.bio || "");
        setBioDraft(data.bio || "");
      }
    } catch (e) {
      console.error("Failed to fetch profile:", e);
    }
  };

  const saveProfile = async () => {
    setSavingProfile(true);
    setProfileSaved(false);
    try {
      const res = await fetch(`${API}/api/profile`, {
        method: "PUT",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ bio: bioDraft }),
      });
      if (res.ok) {
        const data = await res.json();
        setBio(data.bio || "");
        setProfileSaved(true);
        setTimeout(() => setProfileSaved(false), 2500);
      }
    } catch (e) {
      console.error("Failed to save profile:", e);
    } finally {
      setSavingProfile(false);
    }
  };

  const fetchChats = async () => {
    try {
      const res = await fetch(`${API}/api/chats`, { headers: authHeaders() });
      if (res.ok) setChats(await res.json());
    } catch (e) {
      console.error("Failed to fetch chats:", e);
    }
  };

  const fetchMessages = async (chatId) => {
    try {
      const res = await fetch(`${API}/api/chats/${chatId}/messages`, { headers: authHeaders() });
      if (res.ok) {
        const data = await res.json();
        setMessages(data.map(m => ({
          id: m.id,
          role: m.role,
          text: m.content,
          sources: m.sources || []
        })));
      }
    } catch (e) {
      console.error("Failed to fetch messages:", e);
    }
  };

  // ===================== AUTH =====================

  const handleAuth = async (e) => {
    e.preventDefault();
    setAuthError("");
    setIsLoadingAuth(true);
    try {
      if (authMode === "login") {
        const formData = new FormData();
        formData.append("username", email);
        formData.append("password", password);
        const res = await fetch(`${API}/api/auth/login`, { method: "POST", body: formData });
        if (!res.ok) throw new Error((await res.json()).detail || "Login failed");
        const data = await res.json();
        localStorage.setItem("retriva_token", data.access_token);
        setToken(data.access_token);
      } else {
        const res = await fetch(`${API}/api/auth/signup`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });
        if (!res.ok) throw new Error((await res.json()).detail || "Signup failed");
        const formData = new FormData();
        formData.append("username", email);
        formData.append("password", password);
        const loginRes = await fetch(`${API}/api/auth/login`, { method: "POST", body: formData });
        if (loginRes.ok) {
          const data = await loginRes.json();
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

  const handleLogout = () => {
    localStorage.removeItem("retriva_token");
    setToken("");
    setUser(null);
    setChats([]);
    setActive(null);
    setMessages([]);
    setBio("");
    setBioDraft("");
    setShowCitationPanel(false);
  };

  // ===================== CHAT OPERATIONS =====================

  const newChat = async () => {
    try {
      const res = await fetch(`${API}/api/chats`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ title: "New Conversation" }),
      });
      if (res.ok) {
        const c = await res.json();
        setChats((p) => [c, ...p]);
        setActive(c.id);
        setMessages([]);
        setInput("");
        setMobile(false);
      }
    } catch (e) {
      console.error("Failed to create chat:", e);
    }
  };

  const del = async (id) => {
    try {
      await fetch(`${API}/api/chats/${id}`, { method: "DELETE", headers: authHeaders() });
      setChats((p) => p.filter((c) => c.id !== id));
      if (active === id) { setActive(null); setMessages([]); }
      setMenu(null);
    } catch (e) {
      console.error("Failed to delete chat:", e);
    }
  };

  const rename = (c) => { setEditing(c.id); setEdit(c.title); setMenu(null); };

  const save = async (id) => {
    const newTitle = edit.trim() || "New Conversation";
    try {
      await fetch(`${API}/api/chats/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ title: newTitle }),
      });
      setChats((p) => p.map((c) => (c.id === id ? { ...c, title: newTitle } : c)));
    } catch (e) {
      console.error("Failed to rename:", e);
    }
    setEditing(null);
  };

  // ===================== FILE UPLOAD =====================

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || !token) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API}/api/ingest`, {
        method: "POST",
        headers: authHeaders(),
        body: formData,
      });
      const data = await res.json();
      if (res.ok) alert(`Success! Added ${data.chunks} chunks to the knowledge base.`);
      else alert(`Error: ${data.detail}`);
    } catch {
      alert("Failed to upload file.");
    } finally {
      setUploading(false);
      event.target.value = null;
    }
  };

  // ===================== SEND MESSAGE (STREAMING) =====================

  const send = async (raw = input) => {
    const text = raw.trim();
    if (!text || isStreaming || !token) return;

    let chatId = active;

    // Create chat if none active
    if (!chatId) {
      try {
        const res = await fetch(`${API}/api/chats`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...authHeaders() },
          body: JSON.stringify({ title: text.slice(0, 40) }),
        });
        if (res.ok) {
          const c = await res.json();
          chatId = c.id;
          setChats((p) => [c, ...p]);
          setActive(chatId);
        }
      } catch {
        return;
      }
    }

    // Add user message to UI immediately
    const tempUserMsg = { id: `temp-user-${Date.now()}`, role: "user", text, sources: [] };
    setMessages((prev) => [...prev, tempUserMsg]);
    setInput("");
    setIsStreaming(true);

    // Add placeholder assistant message for streaming
    const assistantId = `temp-asst-${Date.now()}`;
    setMessages((prev) => [...prev, { id: assistantId, role: "assistant", text: "", sources: [], isStreaming: true }]);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API}/api/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ query: text, chat_id: chatId, temperature }),
        signal: controller.signal,
      });

      if (!response.ok) {
        if (response.status === 401) { handleLogout(); return; }
        throw new Error("Stream failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullText = "";
      let sources = [];
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.substring(6).trim();
          if (!raw) continue;

          try {
            const parsed = JSON.parse(raw);
            if (parsed.type === "sources") {
              sources = parsed.data;
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, sources } : m))
              );
            } else if (parsed.type === "meta") {
              if (parsed.data?.chat_id && !chatId) {
                chatId = parsed.data.chat_id;
                setActive(chatId);
              }
            } else if (parsed.type === "token") {
              // Tokens already carry their own spacing/newlines.
              fullText += parsed.data || "";
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, text: fullText } : m))
              );
            } else if (parsed.type === "clarification") {
              fullText = parsed.data || fullText;
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, text: fullText } : m))
              );
            } else if (parsed.type === "done") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, isStreaming: false } : m
                )
              );
            } else if (parsed.type === "error") {
              fullText = parsed.data || "An error occurred.";
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, text: fullText, isStreaming: false } : m))
              );
            }
          } catch {
            // Skip malformed JSON lines
          }
        }
      }

    } catch (error) {
      if (error.name === "AbortError") {
        // User pressed Stop: keep whatever was generated so far.
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, text: m.text || "⏹ Stopped.", isStreaming: false }
              : m
          )
        );
      } else {
        console.error("Stream error:", error);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, text: "Sorry, I couldn't connect to the backend.", isStreaming: false }
              : m
          )
        );
      }
    } finally {
      abortRef.current = null;
      setIsStreaming(false);
      fetchChats();
    }
  };

  const stopStreaming = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
  };

  // ===================== CITATION HANDLING =====================

  const handleCitationClick = (citationNum, sources) => {
    const index = parseInt(citationNum) - 1;
    if (sources && sources[index]) {
      setSelectedCitation({ number: citationNum, source: sources[index] });
      setShowCitationPanel(true);
    }
  };

  const formatChunkText = (text) => {
    if (!text) return <span className="text-[#606a66] italic">No content available.</span>;
    return text.split("\n").map((line, index) => {
      if (line.startsWith("## "))
        return <h3 key={index} className="text-base font-bold text-[#e8eceb] mt-4 mb-2 border-b border-[#222a26] pb-1">{line.replace("## ", "")}</h3>;
      if (line.startsWith("# "))
        return <h2 key={index} className="text-lg font-bold text-[#65e6a5] mt-4 mb-2">{line.replace("# ", "")}</h2>;
      if (line.trim() === "") return <br key={index} />;
      return <p key={index} className="mb-2 text-[#d2d9d5]">{line}</p>;
    });
  };

  const renderMessageText = (text, sources) => {
    if (!text) return null;
    return text.split(/(\[\d+\])/g).map((part, idx) => {
      if (/^\[\d+\]$/.test(part)) {
        const citationNum = part.replace(/[\[\]]/g, "");
        return (
          <button key={idx} onClick={() => handleCitationClick(citationNum, sources)}
            className="mx-0.5 inline-flex h-4 min-w-[16px] items-center justify-center rounded bg-[#13251c] px-1 text-[10px] font-bold text-[#65e6a5] hover:bg-[#1a2f26] hover:scale-110 transition-transform cursor-pointer"
            title={`View source ${citationNum}`}>
            {citationNum}
          </button>
        );
      }
      return <span key={idx}>{part}</span>;
    });
  };

  const activeChat = chats.find((c) => c.id === active);

  // ===================== AUTH UI =====================

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
          <h2 className="text-2xl font-bold mb-6 text-center">
            {authMode === "login" ? "Welcome Back" : "Create Account"}
          </h2>
          {authError && (
            <div className="mb-4 p-3 rounded-lg bg-red-900/20 border border-red-800 text-red-400 text-sm text-center">{authError}</div>
          )}
          <form onSubmit={handleAuth} className="space-y-4">
            <div>
              <label className="block text-sm mb-2 text-[#8a9490]">Email</label>
              <input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 bg-[#151a18] border border-[#27302c] rounded-lg outline-none focus:border-[#3d4b44] text-sm" required />
            </div>
            <div>
              <label className="block text-sm mb-2 text-[#8a9490]">Password</label>
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 bg-[#151a18] border border-[#27302c] rounded-lg outline-none focus:border-[#3d4b44] text-sm" required />
            </div>
            <button type="submit" disabled={isLoadingAuth}
              className="w-full py-3 bg-[#13251c] border border-[#31523f] text-[#65e6a5] rounded-lg font-medium hover:bg-[#1a2f26] disabled:opacity-50 flex items-center justify-center gap-2">
              {isLoadingAuth ? "Processing..." : authMode === "login" ? "Sign In" : "Sign Up"}
            </button>
          </form>
          <p className="mt-6 text-center text-sm text-[#65706b]">
            {authMode === "login" ? "Don't have an account? " : "Already have an account? "}
            <button onClick={() => { setAuthMode(authMode === "login" ? "signup" : "login"); setAuthError(""); }}
              className="text-[#65e6a5] hover:underline font-medium">
              {authMode === "login" ? "Sign Up" : "Sign In"}
            </button>
          </p>
        </div>
      </div>
    );
  }

  // ===================== MAIN APP UI =====================

  return (
    <div className="h-screen w-full overflow-hidden bg-[#0b0e0d] text-[#e8eceb] flex">
      {/* Main Content Area */}
      <div className={`flex-1 flex flex-col transition-all duration-300 ease-in-out ${showCitationPanel ? "mr-[450px]" : ""}`}>
        <div className="flex h-full">
          {mobile && <div className="fixed inset-0 z-40 bg-black/60 lg:hidden" onClick={() => setMobile(false)} />}

          {/* Sidebar */}
          <aside className={`fixed inset-y-0 left-0 z-50 flex w-[285px] flex-col border-r border-[#202623] bg-[#101412] transition-transform duration-200 lg:static lg:translate-x-0 ${mobile ? "translate-x-0" : "-translate-x-full"}`}>
            <div className="flex h-[72px] items-center justify-between px-5">
              <div className="flex items-center gap-3">
                <div className="grid h-9 w-9 place-items-center rounded-xl border border-[#31523f] bg-[#13251c]">
                  <Sparkles size={17} className="text-[#65e6a5]" />
                </div>
                <div>
                  <div className="text-[15px] font-semibold">Retriva</div>
                  <div className="text-[10px] uppercase tracking-[.18em] text-[#65706b]">Finance AI</div>
                </div>
              </div>
              <button className="rounded-lg p-2 text-[#6d7773] lg:hidden" onClick={() => setMobile(false)}>
                <X size={18} />
              </button>
            </div>

            <div className="px-3">
              <button onClick={newChat}
                className="flex w-full items-center gap-3 rounded-xl border border-[#27302c] bg-[#151a18] px-4 py-3 text-sm font-medium hover:border-[#3d4b44]">
                <Plus size={17} className="text-[#72e7ad]" /> New chat
              </button>
            </div>

            <div className="px-3 pt-5">
              <div className="relative">
                <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#68716e]" />
                <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search chats"
                  className="w-full rounded-lg bg-[#151a18] py-2.5 pl-9 pr-3 text-xs outline-none placeholder:text-[#606a66] focus:border focus:border-[#2e3c35]" />
              </div>
            </div>

            <div className="mt-4 flex-1 overflow-y-auto px-2">
              <div className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-[.17em] text-[#59625f]">Conversations</div>
              <div className="space-y-1">
                {filtered.map((c) => (
                  <div key={c.id} className="group relative">
                    {editing === c.id ? (
                      <div className="flex rounded-lg bg-[#1b211e] p-1">
                        <input autoFocus value={edit} onChange={(e) => setEdit(e.target.value)}
                          onKeyDown={(e) => { if (e.key === "Enter") save(c.id); if (e.key === "Escape") setEditing(null); }}
                          className="min-w-0 flex-1 bg-transparent px-2 py-1.5 text-xs outline-none" />
                        <button onClick={() => save(c.id)} className="px-2 text-[11px] text-[#72e7ad]">Save</button>
                      </div>
                    ) : (
                      <button onClick={() => { setActive(c.id); setMobile(false); }}
                        className={`flex w-full rounded-lg px-3 py-2.5 text-left text-xs ${active === c.id ? "bg-[#1b211e] text-[#edf2ef]" : "text-[#8a9490] hover:bg-[#171d1a]"}`}>
                        <span className="truncate pr-8">{c.title}</span>
                      </button>
                    )}
                    {editing !== c.id && (
                      <button onClick={(e) => { e.stopPropagation(); setMenu(menu === c.id ? null : c.id); }}
                        className={`absolute right-1.5 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-[#737d79] opacity-0 group-hover:opacity-100 hover:bg-[#29302d] ${menu === c.id ? "opacity-100" : ""}`}>
                        <MoreHorizontal size={15} />
                      </button>
                    )}
                    {menu === c.id && (
                      <div className="absolute right-2 top-[42px] z-20 w-32 rounded-lg border border-[#2a322e] bg-[#171c1a] p-1 shadow-2xl">
                        <button onClick={() => rename(c)}
                          className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-xs text-[#c8cfcc] hover:bg-[#232a27]">
                          <Pencil size={13} /> Rename
                        </button>
                        <button onClick={() => del(c.id)}
                          className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-xs text-[#e48e8e] hover:bg-[#2b2020]">
                          <Trash2 size={13} /> Delete
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="border-t border-[#202623] p-3">
              <button onClick={() => fileInputRef.current.click()} disabled={uploading}
                className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-xs text-[#7d8783] hover:bg-[#171d1a] disabled:opacity-50 mb-1">
                {uploading ? "Processing..." : "📄 Upload PDF"}
              </button>
              <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept=".pdf" className="hidden" />
              <button onClick={() => { setBioDraft(bio); setShowProfile(true); }}
                className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-xs text-[#7d8783] hover:bg-[#171d1a] mb-1">
                <User size={16} /> Profile &amp; Bio
              </button>
              <button onClick={() => setShowSettings(true)}
                className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-xs text-[#7d8783] hover:bg-[#171d1a] mb-1">
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

          {/* Main Chat Area */}
          <main className="relative flex min-w-0 flex-1 flex-col">
            <header className="flex h-[64px] items-center border-b border-[#1c2320] px-4 sm:px-7">
              <button onClick={() => setMobile(true)} className="mr-3 rounded-lg p-2 text-[#8b9591] lg:hidden">
                <Menu size={19} />
              </button>
              <div className="text-sm font-medium text-[#bfc7c3]">{activeChat?.title || "New chat"}</div>
              <div className="ml-auto hidden items-center gap-2 rounded-full border border-[#28312d] bg-[#121715] px-3 py-1.5 text-[10px] text-[#6f7975] sm:flex">
                <span className="h-1.5 w-1.5 rounded-full bg-[#64d99d]" /> Knowledge base connected
              </div>
            </header>

            <div className="flex-1 overflow-y-auto">
              {!active || messages.length === 0 ? (
                <div className="mx-auto flex min-h-full max-w-3xl flex-col items-center justify-center px-5 pb-20">
                  <div className="mb-6 grid h-16 w-16 place-items-center rounded-2xl border border-[#294536] bg-[#13241b]">
                    <Bot size={28} className="text-[#6de3a7]" />
                  </div>
                  <h1 className="text-center text-3xl font-semibold tracking-[-.035em] sm:text-4xl">What can I find for you?</h1>
                  <p className="mt-3 max-w-md text-center text-sm leading-6 text-[#707a76]">
                    Ask Retriva about your financial documents. Answers are grounded in your knowledge base.
                  </p>
                  <div className="mt-9 grid w-full max-w-2xl grid-cols-1 gap-2 sm:grid-cols-2">
                    {suggestions.map((s, i) => (
                      <button key={i} onClick={() => send(s)}
                        className="group rounded-xl border border-[#252d29] bg-[#111614] p-4 text-left text-xs leading-5 text-[#9ba49f] transition hover:-translate-y-0.5 hover:border-[#385044] hover:text-[#d8dfdc]">
                        <div className="mb-2 grid h-7 w-7 place-items-center rounded-lg bg-[#19221e] text-[#6ee0a5]">
                          <FileText size={14} />
                        </div>
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="mx-auto max-w-3xl px-4 pb-36 pt-8 sm:px-7">
                  {messages.map((m) => (
                    <div key={m.id} className={`group mb-8 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                      {m.role === "assistant" ? (
                        <div className="flex max-w-[92%] gap-3 sm:max-w-[82%]">
                          <div className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg border border-[#294536] bg-[#13241b]">
                            <Sparkles size={14} className="text-[#6de3a7]" />
                          </div>
                          <div className="flex-1">
                            <div className="whitespace-pre-wrap break-words rounded-2xl rounded-tl-md border border-[#222a26] bg-[#111614] px-4 py-3.5 text-sm leading-7 text-[#d2d9d5]">
                              {m.isStreaming && !m.text ? (
                                <div className="flex items-center gap-1.5">
                                  <span className="dot-1 h-1.5 w-1.5 rounded-full bg-[#6de3a7]" />
                                  <span className="dot-2 h-1.5 w-1.5 rounded-full bg-[#6de3a7]" />
                                  <span className="dot-3 h-1.5 w-1.5 rounded-full bg-[#6de3a7]" />
                                  <span className="ml-2 text-xs text-[#66716c]">Searching knowledge base…</span>
                                </div>
                              ) : (
                                renderMessageText(m.text, m.sources)
                              )}
                              {m.isStreaming && m.text && (
                                <span className="inline-block w-2 h-4 bg-[#65e6a5] ml-0.5 animate-pulse" />
                              )}
                            </div>
                            {m.sources && m.sources.length > 0 && (
                              <div className="mt-2 flex flex-wrap gap-2">
                                {m.sources.map((s, j) => (
                                  <button key={j} onClick={() => handleCitationClick(j + 1, m.sources)}
                                    className="flex items-center gap-1.5 rounded-full border border-[#29332e] bg-[#131916] px-2.5 py-1.5 text-[10px] text-[#7e8984] hover:border-[#65e6a5] hover:text-[#65e6a5] transition-colors cursor-pointer">
                                    <FileText size={11} className="text-[#63d99e]" />
                                    {s.company} · {s.year}
                                    <span className="text-[#56615c]">· {(s.relevance_score || s.score || 0).toFixed(2)}</span>
                                  </button>
                                ))}
                              </div>
                            )}
                            {!m.isStreaming && m.text && (
                              <button onClick={() => navigator.clipboard.writeText(m.text)}
                                className="mt-2 rounded-md p-1.5 text-[#59635f] opacity-0 transition-opacity hover:bg-[#171d1a] hover:text-[#d2d9d5] group-hover:opacity-100"
                                title="Copy response">
                                <Copy size={13} />
                              </button>
                            )}
                          </div>
                        </div>
                      ) : (
                        <div className="flex max-w-[78%] flex-col items-end sm:max-w-[68%]">
                          <div className="whitespace-pre-wrap break-words rounded-2xl rounded-tr-md bg-[#dfe9e3] px-4 py-3 text-sm leading-6 text-[#17201c]">
                            {m.text}
                          </div>
                          <button onClick={() => navigator.clipboard.writeText(m.text)}
                            className="mt-1 rounded-md p-1.5 text-[#59635f] opacity-0 transition-opacity hover:bg-[#171d1a] hover:text-[#d2d9d5] group-hover:opacity-100"
                            title="Copy message">
                            <Copy size={13} />
                          </button>
                        </div>
                      )}
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Input Bar */}
            <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-[#0b0e0d] via-[#0b0e0d]/95 to-transparent px-4 pb-5 pt-12 sm:px-7">
              <div className="pointer-events-auto mx-auto max-w-3xl">
                <form onSubmit={(e) => { e.preventDefault(); send(); }}
                  className="relative rounded-2xl border border-[#303a35] bg-[#121715] p-2 shadow-2xl focus-within:border-[#46574e]">
                  <textarea ref={ref} rows={1} value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                    placeholder="Ask Retriva about your financial documents..."
                    className="min-h-[44px] w-full resize-none bg-transparent px-3 py-2.5 pr-14 text-sm outline-none placeholder:text-[#59635f]" />
                  {isStreaming ? (
                    <button type="button" onClick={stopStreaming}
                      className="absolute bottom-2.5 right-2.5 grid h-9 w-9 place-items-center rounded-xl bg-[#fbf9f9] text-[#2b1414] hover:opacity-90"
                      title="Stop generating">
                      <Square size={13} fill="currentColor" />
                    </button>
                  ) : (
                    <button type="submit" disabled={!input.trim()}
                      className="absolute bottom-2.5 right-2.5 grid h-9 w-9 place-items-center rounded-xl bg-[#dce9e1] text-[#17201b] disabled:opacity-30"
                      title="Send">
                      <Send size={15} />
                    </button>
                  )}
                </form>
                <div className="mt-2 text-center text-[10px] text-[#4f5955]">
                  Retriva can make mistakes. Verify important financial information.
                </div>
              </div>
            </div>
          </main>
        </div>
      </div>

      {/* ===================== CITATION PANEL ===================== */}
      {showCitationPanel && selectedCitation && (
        <div className="fixed inset-y-0 right-0 w-[450px] bg-[#101412] border-l border-[#202623] shadow-2xl transform transition-transform duration-300 ease-in-out z-50 flex flex-col">
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
            <button onClick={() => setShowCitationPanel(false)}
              className="rounded-lg p-2 text-[#6d7773] hover:bg-[#171d1a] hover:text-white transition-colors">
              <X size={18} />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5">
            <div className="rounded-xl border border-[#222a26] bg-[#0b0e0d] p-4">
              <h4 className="text-sm font-semibold text-[#cbd2cf] mb-3 flex items-center gap-2">
                <FileText size={14} className="text-[#63d99e]" /> Document Metadata
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
                  <span className="font-medium text-[#65e6a5]">
                    {(selectedCitation.source.relevance_score || selectedCitation.source.score || 0).toFixed(3)}
                  </span>
                </div>
              </div>
            </div>
            <div className="rounded-xl border border-[#222a26] bg-[#111614] p-4">
              <h4 className="text-sm font-semibold text-[#cbd2cf] mb-3 flex items-center gap-2">
                <FileText size={14} className="text-[#63d99e]" /> Referenced Content
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
          <div className="border-t border-[#202623] p-4 flex gap-3">
            <button onClick={() => setShowCitationPanel(false)}
              className="flex-1 rounded-lg border border-[#27302c] bg-[#151a18] px-4 py-2.5 text-sm text-[#8a9490] hover:border-[#3d4b44] hover:text-[#e8eceb] transition-colors">
              Close
            </button>
            <button onClick={() => navigator.clipboard.writeText(selectedCitation.source.text || "")}
              className="flex-1 rounded-lg bg-[#13251c] border border-[#31523f] px-4 py-2.5 text-sm text-[#65e6a5] hover:bg-[#1a2f26] transition-colors flex items-center justify-center gap-2">
              <Copy size={14} /> Copy
            </button>
          </div>
        </div>
      )}

      {/* ===================== PROFILE / BIO MODAL ===================== */}
      {showProfile && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-2xl border border-[#202623] bg-[#101412] p-6 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-[#e8eceb]">Profile &amp; Bio</h2>
              <button onClick={() => setShowProfile(false)} className="text-[#6d7773] hover:text-white">
                <X size={20} />
              </button>
            </div>

            <p className="mb-3 text-[11px] leading-5 text-[#65706b]">
              Tell Retriva about you — your name, role, company, preferences.
              This is remembered in <span className="text-[#65e6a5]">every chat</span>.
              Individual chats stay private from each other.
            </p>
            <textarea
              value={bioDraft}
              onChange={(e) => setBioDraft(e.target.value)}
              rows={6}
              maxLength={4000}
              placeholder="e.g. I am Uzair. I work at a software house as a full-stack developer."
              className="w-full resize-none rounded-lg border border-[#27302c] bg-[#151a18] px-3 py-2.5 text-sm outline-none placeholder:text-[#59635f] focus:border-[#3d4b44]"
            />
            <div className="mt-3 flex items-center gap-3">
              <button onClick={saveProfile} disabled={savingProfile}
                className="rounded-lg border border-[#31523f] bg-[#13251c] px-4 py-2 text-xs font-medium text-[#65e6a5] hover:bg-[#1a2f26] disabled:opacity-50">
                {savingProfile ? "Saving…" : "Save Profile"}
              </button>
              {profileSaved && <span className="text-xs text-[#65e6a5]">Saved ✓</span>}
            </div>

            <button onClick={() => setShowProfile(false)}
              className="mt-8 w-full rounded-lg bg-[#13251c] py-2.5 text-sm font-medium text-[#65e6a5] hover:bg-[#1a2f26]">
              Close
            </button>
          </div>
        </div>
      )}

      {/* ===================== SETTINGS MODAL ===================== */}
      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
          <div className="w-96 rounded-2xl border border-[#202623] bg-[#101412] p-6 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-[#e8eceb]">LLM Settings</h2>
              <button onClick={() => setShowSettings(false)} className="text-[#6d7773] hover:text-white">
                <X size={20} />
              </button>
            </div>
            <div className="space-y-6">
              <div>
                <div className="flex justify-between text-xs text-[#8a9490] mb-2">
                  <span>Temperature (Creativity)</span>
                  <span className="text-[#65e6a5]">{temperature.toFixed(2)}</span>
                </div>
                <input type="range" min="0.0" max="1.0" step="0.05" value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))} className="w-full accent-[#65e6a5]" />
                <p className="mt-2 text-[10px] text-[#59625f]">
                  Low (0.1) = Strict &amp; Accurate (Best for Finance). High (0.8) = Creative.
                </p>
              </div>
            </div>
            <button onClick={() => setShowSettings(false)}
              className="mt-8 w-full rounded-lg bg-[#13251c] py-2.5 text-sm font-medium text-[#65e6a5] hover:bg-[#1a2f26]">
              Save Settings
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);