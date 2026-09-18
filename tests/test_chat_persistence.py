"""Run with retriva_env\\Scripts\\python.exe -m unittest discover -s tests."""
import importlib
import json
import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


class ChatPersistenceSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        rag_stub = ModuleType("app.rag_engine")
        rag_stub.RetrivaEngine = Mock()
        rag_stub.RetrivaEngine.return_value.query.return_value = {
            "answer": "Test answer", "sources": [], "time_taken": 0.0
        }
        # Prevent database.py's import-time create_all from touching the real DB.
        previous_rag = sys.modules.get("app.rag_engine")
        sys.modules["app.rag_engine"] = rag_stub
        try:
            with patch("sqlalchemy.create_engine", return_value=cls.engine):
                cls.api = importlib.import_module("api_server")
        finally:
            if previous_rag is None:
                sys.modules.pop("app.rag_engine", None)
            else:
                sys.modules["app.rag_engine"] = previous_rag
        cls.database = importlib.import_module("app.database")
        cls.sessions = sessionmaker(bind=cls.engine)

        def get_test_db():
            with cls.sessions() as db:
                yield db

        cls.api.app.dependency_overrides[cls.database.get_db] = get_test_db
        cls.client = TestClient(cls.api.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.api.app.dependency_overrides.clear()
        cls.engine.dispose()

    def test_auth_chat_lifecycle_and_ownership(self):
        client = self.client
        headers = []
        for email in ("alice@example.test", "bob@example.test"):
            response = client.post("/api/auth/signup", json={
                "email": email, "password": "test-password-123"
            })
            self.assertEqual(response.status_code, 200, response.text)
            response = client.post("/api/auth/login", data={
                "username": email, "password": "test-password-123"
            })
            self.assertEqual(response.status_code, 200, response.text)
            headers.append({"Authorization": "Bearer " + response.json()["access_token"]})
        alice, bob = headers
        self.assertEqual(client.get("/api/users/me", headers=alice).status_code, 200)
        self.assertEqual(client.post("/api/query", headers=alice, json={"query": "Hi"}).status_code, 200)
        self.assertEqual(client.get("/api/chats").status_code, 401)
        self.assertEqual(client.post("/api/chats", json={"title": "No auth", "messages": []}).status_code, 401)
        sources = [{"company": "Example", "text": "Quoted \"text\" — revenue", "score": 0.8}]
        payload = {"title": "Revenue", "messages": [
            {"role": "user", "text": "Revenue?", "sources": "[]"},
            {"role": "assistant", "text": "Answer [1]", "sources": json.dumps(sources)}
        ]}
        response = client.post("/api/chats", headers=alice, json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        saved = response.json()
        chat_id = saved["id"]
        self.assertIsInstance(chat_id, int)
        self.assertEqual(json.loads(saved["messages"][1]["sources"]), sources)
        self.assertEqual(client.get("/api/chats", headers=alice).json(), [saved])
        self.assertEqual(client.get("/api/chats", headers=bob).json(), [])
        self.assertEqual(client.post("/api/chats", headers=bob, json=saved).status_code, 404)
        self.assertEqual(client.delete(f"/api/chats/{chat_id}", headers=bob).status_code, 404)
        saved["title"] = "Renamed"
        saved["messages"] = saved["messages"][:1]
        response = client.post("/api/chats", headers=alice, json=saved)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["title"], "Renamed")
        self.assertEqual(len(response.json()["messages"]), 1)
        with self.sessions() as db:
            self.assertEqual(db.query(self.database.Chat).count(), 1)
            self.assertEqual(db.query(self.database.Message).count(), 1)
        invalid = {**saved, "messages": [{"role": "user", "text": "bad", "sources": "{}"}]}
        self.assertEqual(client.post("/api/chats", headers=alice, json=invalid).status_code, 422)
        self.assertEqual(client.delete(f"/api/chats/{chat_id}").status_code, 401)
        self.assertEqual(client.delete(f"/api/chats/{chat_id}", headers=alice).status_code, 204)
        self.assertEqual(client.get("/api/chats", headers=alice).json(), [])
        self.assertEqual(client.post("/api/chats", headers=alice, json=saved).status_code, 404)
        with self.sessions() as db:
            self.assertEqual(db.query(self.database.Message).count(), 0)


if __name__ == "__main__":
    unittest.main()
