import time

from termcolor import colored


def print_typing(text, color="white"):
    for char in text:
        print(colored(char, color), end="", flush=True)
        time.sleep(0.001)
    print()


def print_verbose(text: str, verbose: bool = False, color: str = "white"):
    if verbose:
        print_typing(text, color)
