import sys
from app.rag_engine import RetrivaEngine

def main():
    print("="*70)
    print(" RETRIVA - Production-Ready Financial AI Assistant")
    print("Type 'quit' or 'exit' to stop.")
    print("="*70)
    
    # Initialize Engine (Takes a minute to load models into memory)
    try:
        engine = RetrivaEngine()
    except Exception as e:
        print(f"Failed to initialize Retriva Engine: {e}")
        sys.exit(1)

    # CLI Loop
    while True:
        try:
            user_input = input("\n You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Shutting down Retriva. Goodbye!")
                break
            
            # Call the engine
            response = engine.query(user_input)
            
            # Print Output
            print(f"\nRetriva: {response['answer']}")
            print(f"️ Time: {response['time_taken']}s | Sources: {len(response['sources'])}")
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"️ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()