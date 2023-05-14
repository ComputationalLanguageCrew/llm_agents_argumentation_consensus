"""
This module contains classes for the implementation of custom dialogue agents
capable of supporting or opposing propositions using available tools, while
maintaining persona.
"""

import re
from typing import List, Optional, Union

from langchain import LLMChain, PromptTemplate
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
    """
    A custom prompt template implementation that formats the conversation history,
    tools, and other necessary information for the language model to understand the
    context.
    """

    template: str
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        """
        Formats the template with the given messages and arguments according to
        the specified context.
        """

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
    """
    A custom output parser implementation that understands the output of language
    model for custom single action agents.
    """

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
    """
    A class that provides an execution environment for an agent with access
    to the language model and tools.
    """

    def __init__(self, llm, memory, tools, verbose=False):
        """
        Initialize the AgentEngine.

        :param llm: the language model instance
        :param memory: shared memory object for the engine
        :param tools: a list of available tools for the agent
        :param verbose: if True, print log messages; otherwise, remain silent
        """
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.verbose = verbose

        self.no_tools_str = ""
        if not self.tools:
            self.no_tools_str = "None"

    def get_tool_names(self):
        """
        Get the list of the names of available tools.

        :return: a list of tool names
        """
        return [tool.name for tool in self.tools]


class Agent:
    """
    A class representing an agent capable of supporting or opposing propositions
    using available tools while maintaining the persona.
    """

    def __init__(self, id: str, name: str, persona, engine: AgentEngine):
        """
        Initialize the Agent with its unique attributes.

        :param id: a unique identifier for the agent
        :param name: the name of the agent
        :param persona: the persona of the agent
        :param engine: the AgentEngine that provides access to the tools and the language model
        """
        self.id = id
        self.name = name
        self.persona = persona
        self.seen_arguments: List[str] = []
        self.argued_arguments: List[str] = []
        self.engine = engine

        template_with_history = f"""
        Pretend that you are {self.name}, a {self.persona}.
        You are taking part in a discussion.
        Support or oppose the given proposition.
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

        Rules: Final answers should be short
        
        
        Previous conversation history:
        {{history}}
        
        Proposition: {{input}}
        
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

        self.executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.engine.tools,
            memory=self.engine.memory,
            verbose=self.engine.verbose,
        )

        voting_template = f"""
        Pretend that you are a {self.persona}.
        Express as best as you can your opinion only with a YES, NO or UNDECIDED 
        about the following argument: {{argument}}"""

        voting_prompt = PromptTemplate(
            template=voting_template, input_variables=["argument"]
        )

        self.vote_chain = LLMChain(
            prompt=voting_prompt,
            llm=self.engine.llm,
            memory=self.engine.memory,
            verbose=self.engine.verbose,
        )

    def __str__(self):
        return f"{self.id}: {self.name}"

    def new_argument(self, result: str, text: str, target_argument: Argument):
        """
        Create a new argument based on the result of the agent's actions.

        :param result: the result of the agent's actions (either support or oppose)
        :param text: the text of the argument
        :param target_argument: the target argument being supported or opposed
        :return: a new Argument object
        """
        new_argument = Argument(
            id=f"{self.id}_{len(self.argued_arguments)+1}",
            text=text,
            creator=self.id,
        )
        if result == "oppose":
            target_argument.opposed_by.append(new_argument.id)
            target_argument.labelling[self.id] = -1
            print(
                f"{self.name} votes argument {target_argument.id}: -1 (by opposing argument)"
            )
        elif result == "support":
            target_argument.supported_by.append(new_argument.id)
            target_argument.labelling[self.id] = 1
            print(
                f"{self.name} votes argument {target_argument.id}: 1 (by supporting argument)"
            )
        return new_argument

    def see_argument(self, argument: Argument):
        """
        Add an argument to the list of seen arguments.

        :param argument: the Argument object seen by the agent
        """
        self.seen_arguments.append(argument.id)

    def argue(self, argument: Argument) -> Optional[Argument]:
        """
        Generate a response for the given argument.

        :param argument: the Argument object to be responded to
        :return: a new Argument object for the generated response or None if no response
        """
        response = self.executor.run(argument.text)

        match = re.match(
            r"^\[(.*?)\]", response
        )  # match [pass/support/oppose] in string

        if not match:
            print("** failure **")  # LLM fail, agent ignores argument
            return None

        print(response)

        result = match.group(1)

        if "pass" in result.lower():
            print("** failure **")
            return None

        justification = response.replace(f"[{result}]", "").strip()
        new_argument = self.new_argument(result, justification, argument)
        self.argued_arguments.append(argument.id)

        return new_argument

    def vote(self, arg: Argument) -> int:
        """
        Vote on the given argument.

        :param arg: the Argument object to be voted on
        :return: 1 for 'yes', -1 for 'no', and 0 for 'undecided'
        """
        label = self.vote_chain.run(arg.text)
        label = "".join(
            char for char in label.lower() if char.isalpha() or char.isspace()
        )

        if "yes" in label:
            return 1
        if "no" in label:
            return -1
        # 'undecided'
        return 0
