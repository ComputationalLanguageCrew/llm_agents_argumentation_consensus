"""This module provides classes and functions to simulate the model
 discussions in a target-oriented discussion framework"""

from typing import List, Dict

from todf.agent import Agent
from todf.argument import Argument
from todf.policies import DiscussionExecutionPolicy
from todf.utils import print_verbose

import networkx as nx


class TODF:
    """Represents a target-oriented discussion framework instance."""

    def __init__(self, target: Argument, agents: List[Agent]):
        self.target = target
        self.agents = agents
        self.arguments: List[Argument] = [target]


def compute_majority_label(argument: Argument):
    """Computes the majority label of an argument.

    :param argument: (Argument) The argument to compute the majority label for.
    :return: (int) The majority label for the argument.
    """
    total_sum = sum(argument.labelling.values())
    if total_sum > 0:
        return 1
    elif total_sum == 0:
        return 0
    else:
        return -1


def compute_support_label(
    argument: Argument, arguments: Dict[str, Argument]
) -> int:
    """Computes the support label for an argument in discussion.

    :param argument: (Argument) The argument to compute the support for.
    :param arguments: (Dict[str, Argument]) A dictionary of arguments in the
                    discussion.
    :return: (int) The computed support value for the argument.
    """
    if len(argument.supported_by) == 0 and len(argument.opposed_by) == 0:
        # leaf
        return compute_majority_label(argument)

    pro = 0
    con = 0

    for child_argument_id in argument.supported_by:
        child_argument = arguments[child_argument_id]
        child_support = compute_support_label(child_argument, arguments)
        if child_support == 1:
            pro += 1
        elif child_support == -1:
            con += 1

    for child_argument_id in argument.opposed_by:
        child_argument = arguments[child_argument_id]
        child_support = compute_support_label(child_argument, arguments)
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
    """Represents a discussion that follows the target-oriented approach,
    involving agents and arguments."""

    def __init__(
        self,
        proposition: Argument,
        agents: List[Agent],
        policy: DiscussionExecutionPolicy,
        summarization_llm,
        verbose: bool = True,
    ):
        self.proposition = proposition
        self.framework = TODF(target=proposition, agents=agents)
        self.execution_policy = policy
        self.verbose = verbose
        self.summarization_llm = summarization_llm

    def __repr__(self):
        return self.proposition

    def argue(self):
        """Executes the discussion."""
        self.execution_policy.exec(
            framework=self.framework, verbose=self.verbose
        )

    def summarize(self):
        # assumption: first argument is the target and arguments are chronologically sorted

        arguments = "\n\n".join(map(lambda arg: arg.creator + ": " + arg.text, self.framework.arguments[1:]))

        topic = self.framework.target.text

        consensus = {1: "YES", 0: "UNDECIDED", -1: "NO"}[self.consensus()]

        prompt = f"""Given the discussion below:

        Topic: {topic}

        {arguments}

        Consensus decision: {consensus}

        Provide a summary of the discussion, presenting the main arguments provided and comment on the consensus reached."""

        return self.summarization_llm(prompt)

    def get_graph(self):
        """
        Returns the discussion graph.

        :return: (nx.DiGraph) The discussion graph.
        """
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
        """
        Saves the discussion graph to a file.

        :param path: (str) The path to save the graph file.
        """
        return nx.write_graphml(self.get_graph(), path)

    def consensus(self) -> int:
        """
        Computes and returns the consensus of the discussion.

        :return: (int) The consensus value (-1, 0, or 1) for the discussion.
        """
        arguments = {}
        for argument in self.framework.arguments:
            arguments[argument.id] = argument

        target = self.framework.target
        return compute_support_label(target, arguments)

    def run(self) -> str:
        """
        Executes the discussion and returns the result

        :return: (str) The discussion result
        """
        self.argue()
        summary = self.summarize()
        return summary
