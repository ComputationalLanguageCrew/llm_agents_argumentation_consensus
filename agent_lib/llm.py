from typing import Dict, List, Tuple, Union

from langchain import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from agent_lib.utils import CustomOutputParser, CustomPromptTemplate


class MissingAgreementStatus(Exception):
    pass

class LLMAgent():

    name: str = None
    executor: AgentExecutor = None
    
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
    
    def argue(self, text: str, defendant: str) -> Dict[str, Union[str, int]]:
        
        label_enc = {
            "agree": 1,
            "disagree": -1,
            "neutral": 0
        }
        
        arg_dict: Dict[str, Union[str, int]] = {
            "defendant": "__name_of_agent__",
            "proposition": "__target__",
            "argument": "__argument__",
            "agreement_state": "__argument__",
            "label": 0,  # 1: agree, -1: disagree, 0: neutral
        }

        arg_dict["defendant"] = defendant
        arg_dict["proposition"] = text
        argument = self.execute(text)
        (
            arg_dict["agreement_state"],
            arg_dict["argument"]
        ) = self.parse_argument(argument)
        if arg_dict["agreement_state"] not in label_enc.keys():
            raise MissingAgreementStatus(f"Agent {self.name} is missing agreement statement!")
        arg_dict["label"] = label_enc[arg_dict["agreement_state"]]
        return arg_dict

    @staticmethod
    def parse_argument(argument: str) -> Tuple[str, str]:
        agreement_state = argument.split(".")[0].strip()
        justification = argument.replace(f"{agreement_state}.", "")
        justification = justification.replace("\n\n", "").replace("\n", "").strip()
        agreement_state = argument.split(".")[0].strip().lower()
        return agreement_state, justification
