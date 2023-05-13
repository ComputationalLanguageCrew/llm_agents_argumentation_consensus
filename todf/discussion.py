import time
from dataclasses import dataclass

from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator
from termcolor import colored


def print_typing(text, color="white"):
    for char in text:
        print(colored(char, color), end="", flush=True)
        time.sleep(0.02)
    print()


@dataclass
class Agent:
    id: str
    name: str
    role: str
    description: str


proposition = "Universal Basic Income is an effective solution to combat poverty and inequality."


agent1 = Agent(
    id="ag1",
    name="Sarah Thompson",
    role="Social Worker",
    description="Sarah is a social worker who has witnessed the struggles of marginalized communities. She supports UBI as a means to uplift individuals out of poverty and provide them with a stable foundation for a better life",
)

agent1 = Agent(
    id="ag1",
    name="Sarah Thompson",
    role="Social Worker",
    description="Has witnessed the struggles of marginalized communities. She supports UBI as a means to uplift individuals out of poverty and provide them with a stable foundation for a better life.",
)

agent2 = Agent(
    id="ag2",
    name="John Reynolds",
    role="Small Business Owner",
    description="Is concerned about the potential impact of UBI on the labor market and his ability to compete. He has reservations about UBI and its potential effects on the entrepreneurial spirit.",
)

agent3 = Agent(
    id="ag3",
    name="Dr. Emily Carter",
    role="Economist",
    description="She has conducted extensive research on UBI and will provide evidence-based arguments and analysis to inform the discussion.",
)

agent4 = Agent(
    id="ag4",
    name="Mark Johnson",
    role="Curious Citizen",
    description="Is interested in learning more about UBI. He has no strong preconceptions about the topic and seeks to gain a better understanding of its potential benefits and drawbacks.",
)

agent5 = Agent(
    id="ag5",
    name="Linda Thompson",
    role="Concerned Taxpayer",
    description="Worries about the financial feasibility of UBI and its potential burden on government budgets. She has limited knowledge of UBI and seeks clarification on its costs and funding mechanisms.",
)

agents = [agent1, agent2, agent3, agent4, agent5]


class ArgumentationFirstRoundPromptTemplate(StringPromptTemplate, BaseModel):
    """A prompt template for the first round of argumentation"""

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        if len(v) != 2 or "agent" not in v or "proposition" not in v:
            raise ValueError("function_name must be the only input_variable.")
        return v

    def format(self, **kwargs) -> str:
        # Get the source code of the function
        agent = kwargs["agent"]
        proposition = kwargs["proposition"]

        # Generate the prompt to be sent to the language model
        prompt = f"""
        You are {agent.name}, a {agent.role}.
        {agent.description}

        You will be given a proposition.

        You must follow the following chain of thought:

        1. Based on my role and expertise I have valuable input and can argue the proposition
        2. My most valuable and insightful argument is ...
        3. My argument attacks/defends the initial proposition

        The reponse format should ALWAYS be:
        '''json
        {{
        'valuable_input':  True/False,
        'argument': <argument>,
        'relation_to_proposition': 'attacks' or 'defends'
        }}
        '''


        proposition: {proposition}
        """
        return prompt

    def _prompt_type(self):
        return "first-round-argumentation"


arg_round_1 = ArgumentationFirstRoundPromptTemplate(
    input_variables=["agent", "proposition"]
)

# Generate a prompt for the function "get_source_code"
prompt = arg_round_1.format(agent=agent1, proposition=proposition)

print_typing("\nDebate proposition:")
print_typing(
    proposition,
    "cyan",
)
print_typing("\nLet's meet our participants:")

for agent in agents:
    print_typing(
        f"{agent.name}\nRole: {agent.role}\nDescription: {agent.description}\n",
        "yellow",
    )
    time.sleep(1)  # Pause between each agent introduction
