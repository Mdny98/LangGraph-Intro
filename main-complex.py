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

# Setting up the state for the graph

'''
a type for stuctred output parser; LLM can find out how to give you a respone in this format

'''
class MessageClassifier(BaseModel):
    # "Literal" :Exact value
    # pydantic field 
    message_type: Literal["emotional", "logical"]= Field (
        ...,
        # "description" is : how we are going to fill-in this value (message type)
        description="Classify if the message requires an emotional (therapist) or logical response"

    )


# state of our model or agent the have access to (control flow of the applicaion)
# It is actually  the type of the information through the graph
class State(TypedDict):

    # this line say: "messages" is a type of "list" and every changes is provided by "add_meesages" 
    # this is for storing messages: user, assistant (LLM) and ...
    messages: Annotated[list, add_messages]
    # this is added for holding the type of the message => emotional or logical
    message_type: str| None

#Graph Builder

# node
def classify_message(state: State):
    last_message = state["messages"][-1]
    # this version of our LLM gives us the output matches with that pydantic model(in MessageClassifer)
    classifier_llm = llm.with_structured_output(MessageClassifier)
    result = classifier_llm.invoke([
        # list of the messages we want to invoke
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """
        },
        {"role": "user", "content": last_message.content}
        # After this state we are able to access to the "message_type" which is matches by "str|None"
    ])
    return {"message_type": result.message_type}

def router(state:State):
    # logical is the defult
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    
    return {"next": "logical"}

def therapist_agent(state:State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": '(em) '+reply.content}]}


def logical_agent(state:State):

    last_message = state["messages"][-1]
    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": '(log) ' + reply.content}]}


graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

# we need conditional edges for multiple edges
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    # Path map
    # We dont have "next" in our state defination but in rouuter returned value we have that and can have access to it
    {"therapist": "therapist", "logical":"logical"}
)
graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)


graph = graph_builder.compile()

# define a function to utilize the functions and graph
def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()