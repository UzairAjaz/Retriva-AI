"""
Retriva command line interface.

Usage:  python main.py
The CLI keeps an in-process conversation history so follow-up questions work,
while the engine also provides cross-chat memory backed by ChromaDB.
"""

import sys

from app.rag_engine import RetrivaEngine


def main() -> None:
    print("=" * 70)
    print(" RETRIVA - Local Financial AI Assistant")
    print(" Type 'quit' or 'exit' to stop.")
    print("=" * 70)

    try:
        engine = RetrivaEngine()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to initialize Retriva Engine: {exc}")
        sys.exit(1)

    history = []
    while True:
        try:
            user_input = input("\n You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Shutting down Retriva. Goodbye!")
                break

            print("\nRetriva: ", end="", flush=True)
            answer_parts = []
            used_rag = False
            for event in engine.stream_query(
                user_input, user_id="cli-user", chat_history=history, chat_id="cli"
            ):
                if event["type"] == "token":
                    text = event["data"]
                    answer_parts.append(text)
                    print(text, end="", flush=True)
                elif event["type"] == "done":
                    used_rag = event["data"].get("used_rag", False)
            print()

            answer = "".join(answer_parts).strip()
            if answer:
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": answer})
                # Persist into cross-chat memory as well.
                engine.memory_add(user_input, "user", "cli-user", "cli")
                if used_rag:
                    engine.memory_add(answer, "assistant", "cli-user", "cli")
            print(f" [{'RAG' if used_rag else 'chat'}]")
            print("-" * 70)

        except KeyboardInterrupt:
            print("\n Interrupted. Goodbye!")
            break
        except Exception as exc:  # noqa: BLE001
            print(f" An unexpected error occurred: {exc}")


if __name__ == "__main__":
    main()
