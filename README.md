# Consensus on LLM Agents argumentation

A simple tool based on Langchain and LLMs.

## Description

Given a starting proposition/statement in natural language (English for now) we let a number of LLM Agents argue on their agreement, disagreement or neutrality on the matter.

This tool makes it possible to calculate a collective decision among the Agents' argumentation process.

## Getting Started

### Dependencies

All necessary python packages and their dependecies are included in files

- Pipfile
- Pipfile.lock

Run

```
pipenv install --dev
```

to create and install a virtual environment for the project.

### Executing program

From project root dir run:

```
pipenv shell
```

to activate pipenv virtual environment

And then run:

```
python argue.py
```

In `argue.py` you can set variable `'proposition'` euqal to whatever statement you wish your agents to argue on.
