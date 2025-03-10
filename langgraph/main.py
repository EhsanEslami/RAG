from colorama import Fore
from graph import query 

def start():
    instructions = (
        """How can I help you today ?\n"""
    )
    print(Fore.BLUE + "\n\x1B[3m" + instructions + "\x1B[0m" + Fore.RESET)

    print("MENU")
    print("====")
    print("[1]- Ask a question")
    print("[2]- Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        ask()
    elif choice == "2":
        print("Goodbye!")
        exit()
    else:
        print("Invalid choice")
        start()


def ask():
    while True:
        user_input = input("Q: ")
        # Exit
        if user_input == "x":
            start()
        else:

            response = query(user_input)
            print(Fore.BLUE + "\n=== ANSWER ====")
            print("A: " + response + Fore.RESET)
            print(Fore.WHITE + 
                  "\n-------------------------------------------------")


if __name__ == "__main__":
    start()
