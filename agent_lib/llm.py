
from dataclasses import dataclass
from typing import Dict, List, Tuple

from langchain import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from agent_lib.utils import CustomOutputParser, CustomPromptTemplate


class MissingAgreementStatus(Exception):
    pass


@dataclass
class Argument:

    id: int
    proposition: str
    proposition_id: int
    argument: str
    agreement_state: str
    label: int
    
    def __init__(self):
        pass
    
    def to_dict(self):
        return {
            "id": self.id,
            "proposition": self.proposition,
            "proposition_id": self.proposition_id,
            "argument": self.argument,
            "agreement_state": self.agreement_state,
            "label": self.label,
        }


class LLMAgent():

    name: str
    executor: AgentExecutor
    
    def __init__(
        self,
        prompt_template: str,
        tools: List[Tool],
        llm_model: str,
        agent_name: str,
    ):
        self.name = agent_name
        
        prompt = CustomPromptTemplate(
            template=prompt_template,
            tools=tools,
            input_variables=["input", "intermediate_steps", "history"]
        )
        output_parser = CustomOutputParser()

        tool_names = [tool.name for tool in tools]

        llm = ChatOpenAI(model_name=llm_model, temperature=0)
        memory = ConversationBufferMemory(memory_key="history")
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )

        self.executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)

        return

    def execute(self, text: str) -> str:
        return self.executor.run(text)
    
    def argue(self, id: str, proposition: str, proposition_id: str) -> Argument:
        
        label_enc = {
            "agree": 1,
            "disagree": -1,
            "neutral": 0
        }
        
        arg = Argument()

        arg.id = id
        arg.proposition_id = proposition_id
        arg.proposition = proposition
        argument = self.execute(proposition)
        (
            arg.agreement_state,
            arg.argument
        ) = self.parse_argument(argument)
        if arg.agreement_state not in label_enc.keys():
            raise MissingAgreementStatus(f"Agent {self.name} is missing agreement statement!")
        arg.label = label_enc[arg.agreement_state]
        return arg

    @staticmethod
    def parse_argument(argument: str) -> Tuple[str, str]:
        agreement_state = argument.split(".")[0].strip()
        justification = argument.replace(f"{agreement_state}.", "")
        justification = justification.replace("\n\n", "").replace("\n", "").strip()
        agreement_state = argument.split(".")[0].strip().lower()
        return agreement_state, justification
