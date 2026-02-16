from src.simpleConversation import ChatState, graph


def main():
    state = ChatState(messages=[])
    print("Chatbot indult. Írj valamit, vagy 'exit'-tel lépj ki.")
    while True:
        user_input = input("Te: ")
        if user_input.lower() == "exit":
            print("Viszlát!")
            break

        state["messages"].append({"role": "user", "content": user_input})

        state = graph.invoke(state)

        lastmessage = state["messages"][-1]
        print(f"Chatbot ({state['last_agent']}):", lastmessage.content[0]["text"])


if __name__ == "__main__":
    main()
