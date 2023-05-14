from typing import List, Optional, Dict

from todf.agent import Agent
from todf.argument import Argument
from todf.policies import DiscussionExecutionPolicy, ExecutionPolicies
from todf.utils import print_verbose

import networkx as nx


class TODF:
    def __init__(self, target: Argument, agents: List[Agent]):
        self.target = target
        self.agents = agents
        self.arguments: List[Argument] = [target]


def compute_majority_label(argument: Argument):
    return sum(argument.labelling.values())


def compute_support(argument: Argument, arguments: Dict[str, Argument]) -> int:
    if len(argument.supported_by) == 0 and len(argument.opposed_by):
        # leaf
        return compute_majority_label(argument)

    pro = 0
    con = 0

    for child_argument_id in argument.supported_by:
        child_argument = arguments[child_argument_id]
        child_support = compute_support(child_argument, arguments)
        if child_support == 1:
            pro += 1
        elif child_support == -1:
            con += 1

    for child_argument_id in argument.opposed_by:
        child_argument = arguments[child_argument_id]
        child_support = compute_support(child_argument, arguments)
        if child_support == 1:
            con += 1
        elif child_support == -1:
            pro += 1

    if pro > con:
        return 1
    elif pro < con:
        return -1
    else:
        return compute_majority_label(argument)


class Discussion:
    def __init__(
        self,
        proposition: Argument,
        agents: List[Agent],
        policy: DiscussionExecutionPolicy,
        verbose: bool = True,
    ):
        self.proposition = proposition
        self.framework = TODF(target=proposition, agents=agents)
        self.execution_policy = policy
        self.verbose = verbose

    def __repr__(self):
        return self.proposition

    def run(self):
        print_verbose("Discussion topic:", self.verbose)
        print_verbose(self.proposition.text, self.verbose, "green")
        print_verbose("Introducing the agents:", self.verbose)
        for a in self.framework.agents:
            print_verbose(a.name, self.verbose, "cyan")
            print_verbose(a.persona, self.verbose, "cyan")

        self.execution_policy.exec(
            framework=self.framework, verbose=self.verbose
        )

    def get_graph(self):
        G = nx.DiGraph()

        # create the nodes
        for argument in self.framework.arguments:
            # initialize support function label with 2
            G.add_node(
                argument.id,
                author=argument.creator,
                text=argument.text,
            )

        # create the edges
        for argument in self.framework.arguments:
            for arg_from in argument.supported_by:
                G.add_edge(arg_from, argument, type="supports", color="green")

            for arg_from in argument.opposed_by:
                G.add_edge(arg_from, argument, type="opposes", color="red")

        return G

    def save(self, path: str):
        return nx.write_graphml(self.get_graph(), path)

    def consensus(self) -> int:
        arguments = {}
        for argument in self.framework.arguments:
            arguments[argument.id] = argument

        target = self.framework.target
        return compute_support(target, arguments)
