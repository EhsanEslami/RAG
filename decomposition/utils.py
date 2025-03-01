from colorama import Fore


def format_qa_pair(question, answer):
    """Pairing and format Q and A"""
    
    formatted_string = ""
    formatted_string += f"{Fore.GREEN}Question: {question}{Fore.RESET}\n{Fore.WHITE}Answer: {answer}\n\n {Fore.RESET}"
    print("=====  QUESTION/ANSWER PAIRS: =====")
    print(formatted_string.strip())
    return formatted_string.strip()


def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()
