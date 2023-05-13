import re
from typing import List, Union

from langchain import LLMChain
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
    Tool,
)
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage

from todf.argument import Argument


class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={
                    "output": llm_output.split("Final Answer:")[-1].strip()
                },
                log=llm_output,
            )
        # Parse out the action and action input
        regex = (
            r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        match = re.search(regex, llm_output, re.DOTALL)

        if not match:
            #                 raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            #             print('***** NO MATCH <<', llm_output, '>>')
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )

        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action,
            tool_input=action_input.strip(" ").strip('"'),
            log=llm_output,
        )


class AgentEngine:
    def __init__(self, llm, memory, tools, verbose=False):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.verbose = verbose

        self.no_tools_str = ""
        if not self.tools:
            self.no_tools_str = "None"

    def get_tool_names(self):
        return [tool.name for tool in self.tools]


class Agent:
    def __init__(self, id: str, name: str, persona, engine: AgentEngine):
        self.id = id
        self.name = name
        self.persona = persona
        self.seen_arguments: List[str] = []
        self.argued_arguments: List[str] = []
        self.engine = engine

    def __str__(self):
        return f"{self.id}: {self.name}"

    def see_argument(self, argument: Argument):
        self.seen_arguments.append(argument.id)

    def argue(self, argument: Argument):
        template_with_history = f"""
        Pretend that you are a {self.persona}. 
        Support or oppose the following proposition: {{input}}
        You have access to the following tools: {self.engine.no_tools_str}

        {{tools}}

        Use the following format:

        Proposition: the input proposition you must support or oppose
        Thought: do I want to argue this proposition?

        If I dont want to argue this proposition:
        Final Answer: [pass]

        If I want to argue this proposition:
        Thought: do I have available tools?
        Action: check to see if I have tools
        Thought: do I need to use a tool?
        Action: decide if I want to use a tool
        
        If there are available tools and you decide to use a tool:
        Thought: I should always think about what to do next
        Action: use one of [{{tool_names}}]
        Action Input: the input to the action based on your judgement
        Observation: based on the result of the  action
         ... (this Thought/Action/Action Input/Observation can repeat 2 times)
        Thought: I will argue the proposition based on the observations
        Final Answer: [oppose or support] and a short conclusive argument for opposing or supporting the proposition
        
        If you do not use a tool:
        Thought: I will support or oppose the proposition based on my existing knowledge and beliefs
        Final Answer: [oppose or support] and a short conclusive argument for opposing or supporting the proposition

        Examples of Final Answers:
        [support] I believe that wealth should be equally distributed because it is more fair.
        [oppose] I think that wealth should not be equally distributed and it should be based on work.
        [pass]
        
        
        Previous conversation history:
        {{history}}

        {{agent_scratchpad}}"""
        prompt_with_history = CustomPromptTemplate(
            template=template_with_history,
            tools=self.engine.tools,
            input_variables=["input", "intermediate_steps", "history"],
        )
        output_parser = CustomOutputParser()
        llm_chain = LLMChain(llm=self.engine.llm, prompt=prompt_with_history)
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=self.engine.get_tool_names(),
        )
        executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.engine.tools,
            memory=self.engine.memory,
            verbose=self.engine.verbose,
        )
        response = executor.run(argument.text)

        match = re.match(
            r"^\[(.*?)\]", response
        )  # match [pass/support/oppose] in string

        if not match:
            print("** failure **")  # LLM fail, agent ignores argument

        print(response)

        result = match.group(1)

        if "pass" in result.lower():
            print("** failure **")

    def vote(self):
        pass

    def clear_memory(self):
        pass
