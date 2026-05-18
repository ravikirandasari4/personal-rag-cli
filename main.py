from src.rag_system import PersonalRAG
import os

def main():
    # Initialize RAG system
    rag = PersonalRAG()
    
    # Create documents and data directories if they don't exist
    os.makedirs("documents", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("Personal RAG System")
    print("==================")
    print("Commands:")
    print("1. 'build' - Build index from documents/")
    print("2. 'load' - Load existing index")
    print("3. 'query <question>' - Ask a question")
    print("4. 'summarize' - Get document summary")
    print("5. 'quit' - Exit")
    print()
    
    while True:
        command = input("Enter command: ").strip()
        
        if command == 'quit':
            break
        elif command == 'build':
            if not os.listdir("documents"):
                print("Please add documents to the 'documents/' directory first.")
                continue
            rag.build_index("documents")
        elif command == 'load':
            rag.load_index()
        elif command.startswith('query '):
            question = command[6:]  # Remove 'query ' prefix
            if question:
                print("\nSearching and generating response...")
                response = rag.query(question)
                print(f"\nAnswer: {response}\n")
            else:
                print("Please provide a question after 'query'")
        elif command == 'summarize':
            print("\nGenerating summary...")
            summary = rag.summarize()
            print(f"\nSummary: {summary}\n")
        else:
            print("Unknown command. Please try again.")

if __name__ == "__main__":
    main()