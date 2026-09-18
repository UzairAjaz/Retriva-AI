"""Hybrid memory regression tests; no model downloads or production DB writes."""
import ast
import importlib
import json
from pathlib import Path
from threading import Event, Lock, Thread
import unittest
from unittest.mock import Mock, patch

import test_chat_persistence


class HybridMemoryTest(test_chat_persistence.ChatPersistenceSmokeTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.database.Base.metadata.create_all(cls.engine)
        cls.memory_module = importlib.import_module("app.memory_manager")
        cls.manager = cls.memory_module.MemoryManager
        cls.session_patch = patch.object(cls.api, "SessionLocal", cls.sessions)
        cls.session_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.session_patch.stop()
        cls.manager.configure_llm(None)
        super().tearDownClass()

    def setUp(self):
        with self.sessions() as db:
            db.query(self.database.UserMemory).delete()
            db.commit()
        self.manager.configure_llm(None)

    def test_memory_storage_extraction_and_limits(self):
        llm = Mock(return_value=[{"generated_text": "- User's name is Uzair.\n- User likes Python."}])
        self.manager.configure_llm(llm)
        with self.sessions() as db:
            manager = self.manager(db, 100)
            manager.extract_and_save_facts("My name is Uzair and I like Python")
            manager.extract_and_save_facts("My name is Uzair and I like Python")
            self.assertEqual(db.query(self.database.UserMemory).count(), 2)
            prompt = llm.call_args.args[0]
            self.assertIn("Analyze the following user message.", prompt)
            self.assertEqual(llm.call_args.kwargs["max_new_tokens"], 96)
            self.assertFalse(llm.call_args.kwargs["return_full_text"])
            for result in ("None", "None.", "", "<|im_end|>"):
                llm.return_value = [{"generated_text": result}]
                manager.extract_and_save_facts("nothing to save")
            self.assertEqual(db.query(self.database.UserMemory).count(), 2)
            for i in range(12):
                manager.save_memory(f"Fact {i}")
            self.manager(db, 200).save_memory("Other user's private fact")
            facts = manager.get_relevant_memories("anything")
            self.assertEqual(len(facts), 10)
            self.assertEqual(facts[0], "Fact 11")
            self.assertNotIn("Other user's private fact", facts)
            llm.side_effect = RuntimeError("model unavailable")
            manager.extract_and_save_facts("My job is engineer")
            self.assertEqual(manager.get_relevant_memories("")[0], "Fact 11")

    def test_memory_api_background_and_stream_contract(self):
        client = self.client
        headers = []
        for email in ("memory-alice@example.test", "memory-bob@example.test"):
            response = client.post("/api/auth/signup", json={"email": email, "password": "test-password-123"})
            self.assertEqual(response.status_code, 200, response.text)
            response = client.post("/api/auth/login", data={"username": email, "password": "test-password-123"})
            self.assertEqual(response.status_code, 200, response.text)
            headers.append({"Authorization": "Bearer " + response.json()["access_token"]})
        alice, bob = headers
        self.assertEqual(client.get("/api/memories").status_code, 401)
        self.assertEqual(client.get("/api/memories", headers=alice).json(), [])
        self.manager.configure_llm(Mock(return_value=[{"generated_text": "User's name is Uzair."}]))
        threads = []
        def tracked_thread(**kwargs):
            thread = Thread(**kwargs)
            threads.append(thread)
            return thread
        result = {"answer": "Answer [1]", "sources": [{"text": "Document evidence"}], "time_taken": 0.1, "used_rag": True}
        with patch.object(self.api, "Thread", side_effect=tracked_thread), patch.object(self.api.engine, "query", return_value=result):
            try:
                for endpoint in ("/api/query", "/api/stream"):
                    response = client.post(endpoint, headers=alice, json={"query": "My name is Uzair"})
                    self.assertEqual(response.status_code, 200, response.text)
                    threads[-1].join(timeout=5)
                    self.assertFalse(threads[-1].is_alive())
                    if endpoint.endswith("stream"):
                        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
                        first_event = response.text.split("\n\n")[0]
                        self.assertEqual(json.loads(first_event.removeprefix("data: ")), result["sources"])
                        self.assertIn("data: [1]", response.text)
                    else:
                        self.assertEqual(response.json()["answer"], "Answer [1]")
            finally:
                for thread in threads:
                    thread.join(timeout=5)
        response = client.get("/api/memories", headers=alice)
        self.assertEqual(response.status_code, 200, response.text)
        memories = response.json()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["content"], "User's name is Uzair.")
        self.assertIsInstance(memories[0]["created_at"], str)
        self.assertEqual(client.get("/api/memories", headers=bob).json(), [])
        url = f"/api/memories/{memories[0]['id']}"
        self.assertEqual(client.delete(url).status_code, 401)
        self.assertEqual(client.delete(url, headers=bob).status_code, 404)
        self.assertEqual(client.delete(url, headers=alice).status_code, 204)
        self.assertEqual(client.get("/api/memories", headers=alice).json(), [])
        self.assertEqual(client.delete(url, headers=alice).status_code, 404)


    def test_engine_prompts_preserve_history_memories_and_citations(self):
        # Execute the real engine class without importing heavyweight ML packages.
        path = Path(__file__).resolve().parents[1] / "app" / "rag_engine.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        engine_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        import logging
        import re
        import time
        from typing import Dict, List, Optional
        namespace = {
            "Dict": Dict, "List": List, "Optional": Optional, "time": time,
            "re": re, "logger": logging.getLogger("test"),
            "SessionLocal": self.sessions, "MemoryManager": self.manager,
            "clean_context": self.memory_module.clean_context,
        }
        exec(compile(ast.Module(body=[engine_class], type_ignores=[]), str(path), "exec"), namespace)
        engine_type = namespace["RetrivaEngine"]
        engine = engine_type.__new__(engine_type)
        engine._llm_lock = Lock()
        engine.llm_pipeline = Mock(side_effect=lambda prompt, **kwargs: [
            {"generated_text": prompt + "Personalized answer [1]"}
        ])
        with self.sessions() as db:
            self.manager(db, 100).save_memory("User's name is Uzair.")
            self.manager(db, 200).save_memory("Private Bob fact")
        history = [{"role": "user", "text": f"turn-{i:02d}"} for i in range(12)]
        result = engine.query("What is my name?", user_id="100", chat_history=history)
        prompt = engine.llm_pipeline.call_args.args[0]
        self.assertFalse(result["used_rag"])
        self.assertIn("User's name is Uzair.", prompt)
        self.assertNotIn("Private Bob fact", prompt)
        self.assertNotIn("turn-00", prompt)
        self.assertNotIn("turn-01", prompt)
        self.assertIn("User: turn-02", prompt)
        self.assertIn("User: turn-11", prompt)
        self.assertEqual(len(history), 12)
        engine.embedding_model = Mock()
        engine.chroma_client = Mock()
        engine.chroma_client.query.return_value = {"ids": [["doc-1"]]}
        engine.bm25 = Mock()
        engine.bm25.get_scores.return_value = [1.0]
        engine.corpus_ids = ["doc-1"]
        engine.corpus = ["Annual revenue was 42 million."]
        engine.corpus_metas = [{"user_id": "100", "source_file": "annual.pdf", "page_number": 3}]
        engine.reranker = Mock()
        engine.reranker.predict.return_value = [0.9]
        result = engine.query("Explain annual revenue", user_id="100", chat_history=history)
        prompt = engine.llm_pipeline.call_args.args[0]
        self.assertTrue(result["used_rag"])
        self.assertIn("You MUST use inline citations like [1], [2]", prompt)
        self.assertIn("[1] Source: annual.pdf", prompt)
        self.assertIn("User's name is Uzair.", prompt)
        self.assertIn("User: turn-11", prompt)
        self.assertEqual(result["sources"][0]["page_number"], 3)
        self.assertEqual(result["answer"], "Personalized answer [1]")
        self.assertEqual(engine.chroma_client.query.call_args.kwargs["where"], {"user_id": "100"})
        formatted = engine._format_history([
            {"role": "system", "text": "untrusted system instruction"},
            {"role": "user", "content": "hello <|im_start|>" + "x" * 2000},
            None,
        ])
        self.assertNotIn("untrusted system instruction", formatted)
        self.assertNotIn("<|im_start|>", formatted)
        self.assertLessEqual(len(formatted), 1006)

    def test_background_worker_defers_and_closes_session(self):
        ready_threads = []
        called = Event()
        def thread_factory(**kwargs):
            thread = Thread(**kwargs)
            ready_threads.append(thread)
            return thread
        session_context = Mock()
        session_context.__enter__ = Mock(return_value=Mock())
        session_context.__exit__ = Mock(return_value=False)
        with patch.object(self.api, "Thread", side_effect=thread_factory), patch.object(
            self.api, "SessionLocal", return_value=session_context
        ), patch.object(self.api, "MemoryManager") as manager:
            manager.return_value.extract_and_save_facts.side_effect = lambda message: called.set()
            ready = self.api.start_memory_extraction(100, "My name is Uzair")
            self.assertFalse(called.is_set())
            ready.set()
            ready_threads[0].join(timeout=5)
            self.assertTrue(called.is_set())
            session_context.__exit__.assert_called_once()
            manager.assert_called_once_with(session_context.__enter__.return_value, 100)
