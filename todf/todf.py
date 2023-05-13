from typing import List

from todf.agent import Agent
from todf.argument import Argument
from todf.policies import DiscussionExecutionPolicy, ExecutionPolicies
from todf.utils import print_verbose


class TODF:
    def __init__(self, target: Argument, agents: List[Agent]):
        self.target = target
        self.agents = agents
        self.arguments: List[Argument] = [target]


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

    def save(self):
        pass

    def concensus(self):
        pass
