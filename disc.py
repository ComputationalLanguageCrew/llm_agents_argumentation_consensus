from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
)

from todf.agent import Agent, AgentEngine
from todf.todf import Argument, Discussion

proposition = Argument(
    id="t",
    text="same-sex marriage should not be allowed",
    is_target=True,
    creator="system",
)

llm = ChatOpenAI(temperature=0.6)
memory = ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=300)

agent_engine = AgentEngine(llm, memory=memory, tools=[], verbose=True)

ag1 = Agent(
    id="ag1",
    name="Bob",
    persona="""40 year old unemployed who believes in maintaining traditional societal hierarchies, 
            fears people from different cultural, ethnic, or religious backgrounds and 
            susceptible to conspiracy theories""",
    engine=AgentEngine(
        llm,
        memory=ConversationSummaryBufferMemory(
            llm=OpenAI(), max_token_limit=300
        ),
        tools=[],
        verbose=True,
    ),
)

ag2 = Agent(
    id="ag2",
    name="Alice",
    persona="""30 years old single female, 
            sharp and analytical thinker, 
            forward-thinking and open to new ideas, 
            and living in New York""",
    engine=AgentEngine(
        llm,
        memory=ConversationSummaryBufferMemory(
            llm=OpenAI(), max_token_limit=300
        ),
        tools=[],
        verbose=True,
    ),
)

ag3 = Agent(
    id="ag3",
    name="George",
    persona="""empathetic and open-minded that believes in social justice, 
            a rational approach to problem-solving, 
            empathy towards disadvantaged or marginalized groups and individuals,
            and a passion for constructive dialogue""",
    engine=AgentEngine(
        llm,
        memory=ConversationSummaryBufferMemory(
            llm=OpenAI(), max_token_limit=300
        ),
        tools=[],
        verbose=True,
    ),
)

agents = [ag1, ag2, ag3]

discussion = Discussion(
    proposition=proposition, agents=agents, policy="SEQUENTIAL", verbose=True
)
discussion.run()
