from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType, load_tools
import src._dotenv

llm = OpenAI(temperature=0.0)

tools = load_tools(["llm-math", "wikipedia"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True)

# question = "I have 20 pairs of red socks but someone buys 12, how many do I have left?"
question = "Which brilliant mind invented socks?"
print(agent(question))
