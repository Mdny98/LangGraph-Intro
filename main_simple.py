from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
# ChatOpenAI
from pydantic import BaseModel , Field
from typing_extensions import TypedDict

#Loading Env Vars
load_dotenv()


llm = init_chat_model(
    model="gemma2-9b-it",
    model_provider="groq"
)

# Setting upo the state for the graph
# state of our model or agent the have access to (control flow of the applicaion)
# It is actually  the type of the information through the graph
class State(TypedDict):

    # this line say: "messages" is a type of "list" and every changes is provided by "add_meesages" 
    # this is for storing messages: user, assistant (LLM) and ...
    messages: Annotated[list, add_messages]

#Graph Builder

graph_builder = StateGraph(State)


#Building nodes

'''
Nodes Modify/check the state 
'''
#chatbot (like a function)
def chatbot(state: State):
    return{"messages": [llm.invoke(state["messages"])]}


# using the node => register to the graph
graph_builder.add_node("chatbot", chatbot)

#Always need "start" and "End" Node
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot", END)

#Run the Graph

graph = graph_builder.compile()

user_input = input("Enter a message:")

# pass the state to the graph (same type of "State", TypedDict)
state = graph.invoke({"messages": [{"role":"user","content": user_input}]})


'''
 if you print state["messages"] you will see a lot of fields all of the user and AI messages
'''
print(state["messages"][-1].content)

"drawing the graph"

# uncomment these lines
# from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass