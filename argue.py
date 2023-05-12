import json
from typing import List

from agent_lib.llm import LLMAgent
from agents.liberal_rich_agent import liberal_rich_agent
from agents.middle_class_agent import middle_class_agent
from agents.working_class_agent import working_class_agent
from discussion_framework.discussion import DiscussionFramework

proposition = "A coalition greek government between the big political powers can prove a good choice for the greek people."

agents: List[LLMAgent] = [
    liberal_rich_agent,
    middle_class_agent,
    working_class_agent,
]
max_depth = 2

df = DiscussionFramework(agents)

d = df.argue(proposition, max_depth=max_depth)

