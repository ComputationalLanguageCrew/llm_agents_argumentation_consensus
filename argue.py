import random
from typing import Dict, List

from termcolor import colored

from agent_lib.llm import LLMAgent
from agents.liberal_rich_agent import liberal_rich_agent
from agents.middle_class_agent import middle_class_agent
from agents.working_class_agent import working_class_agent

proposition = "A coalition greek government between the big political powers can prove a good choice for the greek people."

agents: List[LLMAgent] = [
    liberal_rich_agent,
    middle_class_agent,
    working_class_agent,
]

depth = 0
discussion = {}
max_depth = 2

print(colored(f"Init target: {proposition}", "green"))
while True:
    random.shuffle(agents)
    # Let agents argue on the original target
    if depth == 0:
        discussion[depth] = {}
        for agent in agents:
            discussion[depth][agent.name] = []
            print(colored(f"Agent : {agent.name} ", "yellow"))
            print(colored(f"On proposition from : original_proposition'", "red"))
            try:
                arg = agent.argue(
                    text=proposition,
                    defendant="original_proposition"
                )
            except Exception as e:
                raise Exception from e

            discussion[depth][agent.name].append(arg)
        depth += 1
        continue

    # Gather arguments made by agents on the previous discussion depth level
    prev_arguments: Dict[str, List[str]] = {}
    for agent in agents:
        prev_arguments[agent.name] = [
            value for arg in discussion[depth-1][agent.name]
            for key, value in arg.items()
            if key == "argument"
        ]
    
    # Let agents argue on other agents' arguments
    discussion[depth] = {}
    for agent in agents:
        discussion[depth][agent.name] = []
        for prev_name, prev_args in prev_arguments.items():
            if agent.name == prev_name:
                continue
            for prev_arg in prev_args:
                print(colored(f"Agent : {agent.name}", "yellow"))
                print(colored(f"On proposition from : {prev_name} ", "red"))
                try:
                    arg = agent.argue(
                        text=prev_arg,
                        defendant=prev_name
                    )
                except Exception as e:
                    raise Exception from e

                discussion[depth][agent.name].append(arg)
    if depth >= max_depth:
        break
    depth += 1 
