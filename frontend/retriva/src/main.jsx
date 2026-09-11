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
} from "lucide-react";
import "./index.css";

const suggestions = [
  "What are the risks associated with Meta's international operations?",
  "What is Apple's gross margin percentage for 2023?",
  "How does seasonality affect Amazon's business?",
  "What is the effective tax rate for Alphabet in 2023?",
];

function App() {
  const [chats, setChats] = useState([]);
  const [active, setActive] = useState(null);
  const [input, setInput] = useState("");
  const [search, setSearch] = useState("");
  const [mobile, setMobile] = useState(false);
  const [editing, setEditing] = useState(null);
  const [edit, setEdit] = useState("");
  const [menu, setMenu] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const ref = useRef(null);
  const chat = chats.find((c) => c.id === active);
  
  const filtered = useMemo(
    () => chats.filter((c) => c.title.toLowerCase().includes(search.toLowerCase())),
    [chats, search]
  );

  useEffect(() => {
    if (active) ref.current?.focus();
  }, [active]);

  const newChat = () => {
    const id = Date.now();
    setChats((p) => [{ id, title: "New conversation", messages: [] }, ...p]);
    setActive(id);
    setInput("");
    setMobile(false);
  };

  const del = (id) => {
    setChats((p) => p.filter((c) => c.id !== id));
    if (active === id) setActive(null);
    setMenu(null);
  };

  const rename = (c) => {
    setEditing(c.id);
    setEdit(c.title);
    setMenu(null);
  };

  const save = (id) => {
    setChats((p) =>
      p.map((c) =>
        c.id === id ? { ...c, title: edit.trim() || "New conversation" } : c
      )
    );
    setEditing(null);
  };

  const send = async (raw = input) => {
    const text = raw.trim();
    if (!text || loading) return;

    let id = active;
    if (!id) {
      id = Date.now();
      setChats((p) => [
        {
          id,
          title: text.slice(0, 32) + (text.length > 32 ? "..." : ""),
          messages: [],
        },
        ...p,
      ]);
      setActive(id);
    }

    setChats((p) =>
      p.map((c) =>
        c.id === id
          ? {
              ...c,
              title: c.messages.length ? c.title : text.slice(0, 32) + (text.length > 32 ? "..." : ""),
              messages: [...c.messages, { role: "user", text }],
            }
          : c
      )
    );

    setInput("");
    setLoading(true);

    try {
      // REAL API CALL TO PYTHON BACKEND
      const response = await fetch("http://localhost:8000/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });

      if (!response.ok) throw new Error("Backend error");

      const data = await response.json();

      setChats((p) =>
        p.map((c) =>
          c.id === id
            ? {
                ...c,
                messages: [
                  ...c.messages,
                  {
                    role: "assistant",
                    text: data.answer,
                    sources: data.sources || [],
                  },
                ],
              }
            : c
        )
      );
    } catch (error) {
      console.error("Error:", error);
      setChats((p) =>
        p.map((c) =>
          c.id === id
            ? {
                ...c,
                messages: [
                  ...c.messages,
                  {
                    role: "assistant",
                    text: "Sorry, I couldn't connect to the backend. Please make sure `python api_server.py` is running in another terminal.",
                    sources: [],
                  },
                ],
              }
            : c
        )
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen w-full overflow-hidden bg-[#0b0e0d] text-[#e8eceb]">
      <div className="flex h-full">
        {mobile && (
          <div className="fixed inset-0 z-40 bg-black/60 lg:hidden" onClick={() => setMobile(false)} />
        )}
        
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
            <button onClick={newChat} className="flex w-full items-center gap-3 rounded-xl border border-[#27302c] bg-[#151a18] px-4 py-3 text-sm font-medium hover:border-[#3d4b44]">
              <Plus size={17} className="text-[#72e7ad]" />
              New chat
            </button>
          </div>

          <div className="px-3 pt-5">
            <div className="relative">
              <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#68716e]" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search chats"
                className="w-full rounded-lg bg-[#151a18] py-2.5 pl-9 pr-3 text-xs outline-none placeholder:text-[#606a66] focus:border focus:border-[#2e3c35]"
              />
            </div>
          </div>

          <div className="mt-4 flex-1 overflow-y-auto px-2">
            <div className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-[.17em] text-[#59625f]">Conversations</div>
            <div className="space-y-1">
              {filtered.map((c) => (
                <div key={c.id} className="group relative">
                  {editing === c.id ? (
                    <div className="flex rounded-lg bg-[#1b211e] p-1">
                      <input
                        autoFocus
                        value={edit}
                        onChange={(e) => setEdit(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") save(c.id);
                          if (e.key === "Escape") setEditing(null);
                        }}
                        className="min-w-0 flex-1 bg-transparent px-2 py-1.5 text-xs outline-none"
                      />
                      <button onClick={() => save(c.id)} className="px-2 text-[11px] text-[#72e7ad]">Save</button>
                    </div>
                  ) : (
                    <button
                      onClick={() => { setActive(c.id); setMobile(false); }}
                      className={`flex w-full rounded-lg px-3 py-2.5 text-left text-xs ${active === c.id ? "bg-[#1b211e] text-[#edf2ef]" : "text-[#8a9490] hover:bg-[#171d1a]"}`}
                    >
                      <span className="truncate pr-8">{c.title}</span>
                    </button>
                  )}
                  {editing !== c.id && (
                    <button
                      onClick={(e) => { e.stopPropagation(); setMenu(menu === c.id ? null : c.id); }}
                      className={`absolute right-1.5 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-[#737d79] opacity-0 group-hover:opacity-100 hover:bg-[#29302d] ${menu === c.id ? "opacity-100" : ""}`}
                    >
                      <MoreHorizontal size={15} />
                    </button>
                  )}
                  {menu === c.id && (
                    <div className="absolute right-2 top-[42px] z-20 w-32 rounded-lg border border-[#2a322e] bg-[#171c1a] p-1 shadow-2xl">
                      <button onClick={() => rename(c)} className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-xs text-[#c8cfcc] hover:bg-[#232a27]">
                        <Pencil size={13} /> Rename
                      </button>
                      <button onClick={() => del(c.id)} className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-xs text-[#e48e8e] hover:bg-[#2b2020]">
                        <Trash2 size={13} /> Delete
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="border-t border-[#202623] p-3">
            <button className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-xs text-[#7d8783] hover:bg-[#171d1a]">
              <Settings size={16} /> Settings
            </button>
            <div className="mt-1 flex items-center gap-3 px-3 py-2">
              <div className="grid h-7 w-7 place-items-center rounded-full bg-[#26312b] text-[10px]">U</div>
              <div>
                <div className="text-xs font-medium text-[#cbd2cf]">Retriva User</div>
                <div className="text-[10px] text-[#606a66]">AI workspace</div>
              </div>
              <ChevronDown size={14} className="ml-auto text-[#606a66]" />
            </div>
          </div>
        </aside>

        <main className="relative flex min-w-0 flex-1 flex-col">
          <header className="flex h-[64px] items-center border-b border-[#1c2320] px-4 sm:px-7">
            <button onClick={() => setMobile(true)} className="mr-3 rounded-lg p-2 text-[#8b9591] lg:hidden">
              <Menu size={19} />
            </button>
            <div className="text-sm font-medium text-[#bfc7c3]">{chat?.title || "New chat"}</div>
            <div className="ml-auto hidden items-center gap-2 rounded-full border border-[#28312d] bg-[#121715] px-3 py-1.5 text-[10px] text-[#6f7975] sm:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-[#64d99d]" />
              Knowledge base connected
            </div>
          </header>

          <div className="flex-1 overflow-y-auto">
            {!chat || chat.messages.length === 0 ? (
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
                    <button
                      key={i}
                      onClick={() => send(s)}
                      className="group rounded-xl border border-[#252d29] bg-[#111614] p-4 text-left text-xs leading-5 text-[#9ba49f] transition hover:-translate-y-0.5 hover:border-[#385044] hover:text-[#d8dfdc]"
                    >
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
                {chat.messages.map((m, i) => (
                  <div key={i} className={`mb-8 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                    {m.role === "assistant" ? (
                      <div className="flex max-w-[92%] gap-3 sm:max-w-[82%]">
                        <div className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg border border-[#294536] bg-[#13241b]">
                          <Sparkles size={14} className="text-[#6de3a7]" />
                        </div>
                        <div>
                          <div className="rounded-2xl rounded-tl-md border border-[#222a26] bg-[#111614] px-4 py-3.5 text-sm leading-7 text-[#d2d9d5]">
                            {m.text}
                          </div>
                          {m.sources && m.sources.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-2">
                              {m.sources.map((s, j) => (
                                <div key={j} className="flex items-center gap-1.5 rounded-full border border-[#29332e] bg-[#131916] px-2.5 py-1.5 text-[10px] text-[#7e8984]">
                                  <FileText size={11} className="text-[#63d99e]" />
                                  {s.company} · {s.year} · {s.type}
                                  <span className="text-[#56615c]">· {parseFloat(s.relevance_score || s.score).toFixed(2)}</span>
                                </div>
                              ))}
                            </div>
                          )}
                          <button className="mt-2 rounded-md p-1.5 text-[#59635f] hover:bg-[#171d1a]" title="Copy">
                            <Copy size={13} />
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="max-w-[78%] rounded-2xl rounded-tr-md bg-[#dfe9e3] px-4 py-3 text-sm leading-6 text-[#17201c] sm:max-w-[68%]">
                        {m.text}
                      </div>
                    )}
                  </div>
                ))}
                {loading && (
                  <div className="mb-8 flex justify-start">
                    <div className="flex gap-3">
                      <div className="grid h-7 w-7 place-items-center rounded-lg border border-[#294536] bg-[#13241b]">
                        <Sparkles size={14} className="text-[#6de3a7]" />
                      </div>
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
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  send();
                }}
                className="relative rounded-2xl border border-[#303a35] bg-[#121715] p-2 shadow-2xl focus-within:border-[#46574e]"
              >
                <textarea
                  ref={ref}
                  rows={1}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      send();
                    }
                  }}
                  placeholder="Ask Retriva about your financial documents..."
                  className="min-h-[44px] w-full resize-none bg-transparent px-3 py-2.5 pr-14 text-sm outline-none placeholder:text-[#59635f]"
                />
                <button
                  disabled={!input.trim() || loading}
                  className="absolute bottom-2.5 right-2.5 grid h-9 w-9 place-items-center rounded-xl bg-[#dce9e1] text-[#17201b] disabled:opacity-30"
                >
                  <Send size={15} />
                </button>
              </form>
              <div className="mt-2 text-center text-[10px] text-[#4f5955]">
                Retriva can make mistakes. Verify important financial information.
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);