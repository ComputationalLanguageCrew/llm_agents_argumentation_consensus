"""This module provides various execution policies for discussions in a target-oriented discussion framework"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

from todf.utils import print_verbose

if TYPE_CHECKING:
    from todf.todf import TODF


class DiscussionExecutionPolicy(ABC):
    """An abstract class base for all execution policies

    Attributes:
    name (str): A string representing the name of the execution policy.
    description (str): A string representing the description of the execution policy.
    """

    name: str
    description: str

    @abstractmethod
    def exec(self, framework: TODF, verbose: bool = False):
        """
        Executes the discussion following a specific policy.

        :param framework: (TODF) The target-oriented discussion framework instance.
        :param verbose: (bool, optional) Indicates whether to print verbose output. Default is False.
        """
        pass


class ExecutionPolicies:
    """Execution Policies Registry A registry class to manage execution policy subclasses."""

    _registry: Dict = {}

    @classmethod
    def register(cls, execution_policy_cls):
        cls._registry[execution_policy_cls.name] = execution_policy_cls

    @classmethod
    def get_policy(cls, name):
        policy = cls._registry.get(name)

        if policy is None:
            raise KeyError("Invalid execution policy")

        return policy

    @classmethod
    def get_all_names(cls):
        return list(cls._registry.values())


def register_execution_policy(subclass):
    ExecutionPolicies.register(subclass)
    return subclass


@register_execution_policy
class SequentialExecutionPolicy(DiscussionExecutionPolicy):
    """Sequential Execution Policy Executes discussions in a sequential order."""

    name = "SEQUENTIAL"
    description = "Executes discussions in a sequential order."

    def __init__(self, max_arguments: int = 10):
        self.max_arguments = max_arguments

    def exec(self, framework: TODF, verbose: bool = False):
        """Executes the discussion sequentially.

        :param framework: (TODF) The target-oriented discussion framework instance.
        :param verbose: (bool, optional) Indicates whether to print verbose output. Default is False.
        """
        print_verbose("Discussion topic:", verbose)
        print_verbose(framework.target.text, verbose, "green")
        print_verbose("Introducing the agents:", verbose)
        for a in framework.agents:
            print_verbose(f"\n{a.name}", verbose=verbose, color="cyan")
            print_verbose(a.persona, verbose, "cyan")

        print_verbose(
            "\nExecuting discussion sequentially. Argumentation phase is starting...",
            verbose,
        )

        # Argumentation
        random.shuffle(framework.agents)
        has_new_arguments = True
        while (
            len(framework.arguments) < self.max_arguments and has_new_arguments
        ):
            has_new_arguments = False
            for agent in framework.agents:
                for argument in framework.arguments:
                    if (
                        agent.id == argument.creator
                        or argument in agent.seen_arguments
                    ):
                        continue
                    print_verbose(
                        f"\nAgent {agent} sees argument {argument}",
                        verbose=verbose,
                        color="green",
                    )
                    agent.see_argument(argument)
                    new_arg = agent.argue(argument=argument)
                    if new_arg is None:
                        continue
                    else:
                        has_new_arguments = True
                    framework.arguments.append(new_arg)

                    if len(framework.arguments) >= self.max_arguments:
                        break

        # Labelling/Voting
        random.shuffle(framework.agents)
        for agent in framework.agents:
            for arg in framework.arguments:
                # if agent has already a label in that argument continue
                # on counter argument generation, a label is instantly applied
                # depending if it supports or opposes
                if agent.id in arg.labelling.keys():
                    print_verbose(
                        f"\nAgent {agent} already voted argument {arg.id} : {arg.labelling[agent.id]}",
                        verbose=verbose,
                        color="yellow",
                    )
                    continue

                arg.labelling[agent.id] = agent.vote(arg)
                print_verbose(
                    f"\nAgent {agent} voted argument {arg.id} : {arg.labelling[agent.id]}",
                    verbose=verbose,
                    color="yellow",
                )


@register_execution_policy
class RoundExecutionPolicy(DiscussionExecutionPolicy):
    """Round Execution Policy Executes discussions in rounds."""

    name = "ROUNDS"
    description = "Executes discussions in rounds."

    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth

    def exec(self, framework: TODF, verbose: bool = False):
        """
        Executes the discussion in rounds.

        :param framework: (TODF) The target-oriented discussion framework instance.
        :param verbose: (bool, optional) Indicates whether to print verbose output. Default is False.
        """

        print_verbose("Discussion topic:", verbose)
        print_verbose(framework.target.text, verbose, "green")
        print_verbose("\nIntroducing the agents:", verbose)
        for a in framework.agents:
            print_verbose(f"\n{a.name}", verbose=verbose, color="cyan")
            print_verbose(a.persona, verbose, "cyan")

        print_verbose(
            "\nExecuting discussion in rounds. Argumentation phase is starting...",
            verbose=verbose,
        )

        depth = 0
        discussion: Dict[int, Dict] = {}

        # Argumentation
        while depth <= self.max_depth:
            random.shuffle(framework.agents)
            if depth == 0:
                discussion[depth] = {}
                proposition = framework.arguments[0]
                for agent in framework.agents:
                    discussion[depth][agent.id] = []
                    print_verbose(
                        f"\nAgent {agent} sees argument {proposition}",
                        verbose=verbose,
                        color="green",
                    )
                    agent.see_argument(proposition)
                    new_arg = agent.argue(argument=proposition)
                    if new_arg is None:
                        continue
                    framework.arguments.append(new_arg)
                    discussion[depth][agent.id].append(new_arg)
                depth += 1
                continue

            # Let agents argue on other agents' arguments
            discussion[depth] = {}
            for agent in framework.agents:
                discussion[depth][agent.id] = []
                for prev_id, prev_args in discussion[depth - 1].items():
                    if agent.id == prev_id:
                        continue
                    for prev_arg in prev_args:
                        print_verbose(
                            f"\nAgent {agent} sees argument {prev_arg}",
                            verbose=verbose,
                            color="green",
                        )
                        agent.see_argument(prev_arg)
                        new_arg = agent.argue(argument=prev_arg)
                        if new_arg is None:
                            continue
                        framework.arguments.append(new_arg)
                        discussion[depth][agent.id].append(new_arg)

            depth += 1

        # Labelling/Voting
        random.shuffle(framework.agents)
        for agent in framework.agents:
            for arg in framework.arguments:
                # if agent has already a label in that argument continue
                if agent.id in arg.labelling.keys():
                    print_verbose(
                        f"\nAgent {agent} already voted argument {arg.id} : {arg.labelling[agent.id]}",
                        verbose,
                        color="yellow",
                    )
                    continue
                arg.labelling[agent.id] = agent.vote(arg)
                print_verbose(
                    f"\nAgent {agent} voted argument {arg.id} : {arg.labelling[agent.id]}",
                    verbose=verbose,
                    color="yellow",
                )


@register_execution_policy
class RandomExecutionPolicy(DiscussionExecutionPolicy):
    name = "RANDOM"
    description = "Executes discussions randomly."

    @classmethod
    def exec(cls, framework: TODF, verbose: bool = False):
        print("Executing discussion randomly.")
