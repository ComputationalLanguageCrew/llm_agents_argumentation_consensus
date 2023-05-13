from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

from todf.utils import print_verbose

if TYPE_CHECKING:
    from todf.todf import TODF


class DiscussionExecutionPolicy(ABC):
    name: str
    description: str

    @abstractmethod
    def exec(cls, framework: TODF, verbose: bool = False):
        pass


class ExecutionPolicies:
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
    name = "SEQUENTIAL"
    description = "Executes discussions in a sequential order."

    @classmethod
    def exec(cls, framework: TODF, verbose: bool = False):
        if verbose:
            print(
                "Executing discussion sequentially. Argumentation phase is starting..."
            )

        for agent in framework.agents:
            for argument in framework.arguments:
                print_verbose(
                    f"Agent {agent} sees argument {argument}", verbose=verbose
                )
                agent.see_argument(argument)
                agent.argue(argument=argument)


@register_execution_policy
class RoundExecutionPolicy(DiscussionExecutionPolicy):
    name = "ROUNDS"
    description = "Executes discussions in rounds."

    @classmethod
    def exec(cls, framework: TODF, verbose: bool = False):
        print("Executing discussion in rounds.")


@register_execution_policy
class RandomExecutionPolicy(DiscussionExecutionPolicy):
    name = "RANDOM"
    description = "Executes discussions randomly."

    @classmethod
    def exec(cls, framework: TODF, verbose: bool = False):
        print("Executing discussion randomly.")


"""

# Example usage:
seq_policy = ExecutionPolicyRegistry.get_policy(
    "sequential_execution_policy"
)()
print(seq_policy.name)  # Output: sequential_execution_policy
print(
    seq_policy.description
)  # Output: Executes discussions in a sequential order.
seq_policy.exec()

round_policy = ExecutionPolicyRegistry.get_policy("round_execution_policy")()
print(round_policy.name)  # Output: round_execution_policy
print(round_policy.description)  # Output: Executes discussions in rounds.
round_policy.exec()

random_policy = ExecutionPolicyRegistry.get_policy("random_execution_policy")()
print(random_policy.name)  # Output: random_execution_policy
print(random_policy.description)  # Output: Executes discussions randomly.
random_policy.exec()

# Get all policies
all_policies = ExecutionPolicyRegistry.get_policies()
for policy in all_policies:
    print(
        policy.name
    )  # Output: sequential_execution_policy, round_execution_policy, random_execution_policy
    print(
        policy.description
    )  # Output: Executes discussions in a sequential order., Executes discussions in rounds., Executes discussions randomly.
"""
