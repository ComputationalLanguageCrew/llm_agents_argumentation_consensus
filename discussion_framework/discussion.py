import random
from typing import Dict, List

from termcolor import colored

from agent_lib.llm import Argument, LLMAgent


class DiscussionFramework():

    agents: List[LLMAgent]

    def __init__(self, agents: List[LLMAgent]):
        self.agents = agents
        return

    def argue(
        self,
        proposition: str,
        max_depth: int = 1
    ) -> Dict[int, Dict[str, Argument]]:

        depth = 0
        discussion = {}

        print(colored(f"Init target: {proposition}", "green"))
        while depth <= max_depth:
            random.shuffle(self.agents)
            # Let agents argue on the original target
            if depth == 0:
                discussion[depth] = {}
                for i, agent in enumerate(self.agents):
                    discussion[depth][agent.name] = []
                    print(colored(f"Agent : {agent.name} ", "yellow"))
                    print(colored(f"On initial proposition", "red"))
                    try:
                        arg = agent.argue(
                            id=f"{depth}_{i}",
                            proposition=proposition,
                            proposition_id="0"
                        )
                    except Exception as e:
                        raise Exception from e

                    discussion[depth][agent.name].append(arg)
                depth += 1
                continue
            
            # Let agents argue on other agents' arguments
            discussion[depth] = {}
            for i, agent in enumerate(self.agents):
                discussion[depth][agent.name] = []
                for prev_name, prev_args in discussion[depth-1].items():
                    if agent.name == prev_name:
                        continue
                    for j, prev_arg in enumerate(prev_args):
                        print(colored(f"Agent : {agent.name}", "yellow"))
                        print(colored(f"On proposition with id : {prev_arg.id} ", "red"))
                        try:
                            arg = agent.argue(
                                id=f"{depth}_{i}_{j}",
                                proposition=prev_arg.proposition,
                                proposition_id=prev_arg.id,
                            )
                        except Exception as e:
                            raise Exception from e

                        discussion[depth][agent.name].append(arg)
            depth += 1
            
            return discussion