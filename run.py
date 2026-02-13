from src.simpleConversation import ChatState, graph


def content_to_text(content):
    """Üzenet tartalom kinyerése (string, dict vagy list formátumból)."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict) and "text" in content:
        return content["text"]
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        ).strip() or str(content)
    return str(content)


def main():
    state = ChatState(messages=[])
    print("Chatbot indult. Írj valamit, vagy 'exit'-tel lépj ki.")
    while True:
        user_input = input("Te: ")
        if user_input.lower() == "exit":
            print("Viszlát!")
            break

        # Hozzáadjuk a felhasználó üzenetét
        state["messages"].append({"role": "user", "content": user_input})

        # Lefuttatjuk a graphot a jelenlegi állapottal
        state = graph.invoke(state)

        # Kiírjuk a chatbot válaszát (az utolsó üzenetet; LangChain message objektum)
        last = state["messages"][-1]
        content = last.content if hasattr(last, "content") else last
        print("Chatbot:", content_to_text(content))


if __name__ == "__main__":
    main()
