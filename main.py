# Tech stack: Milvus Lite, conversational buffer memory, literal ai

from operator import itemgetter
import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_milvus import Milvus 

from chainlit.types import ThreadDict
import chainlit as cl

from literalai import LiteralClient
import langsmith

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY= os.getenv("LANGSMITH_API_KEY")
LITERAL_API_KEY = os.getenv("LITERAL_API_KEY")


#Important Chatbot Variables
text_splitter = RecursiveCharacterTextSplitter()
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

#Initialize Milvus URI
URI = "./db/milvus_cozmos.db"

# Initialize Literal AI
lai = LiteralClient(
    api_key = LITERAL_API_KEY
)
lai.instrument_openai()

# Initialize LangSmith client
client = langsmith.Client(api_key=LANGSMITH_API_KEY)

# Handler
def process_pdf(file_path: str):
    docs = []
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs


def get_docsearch_from_pdf(file_path: str):
    docs = process_pdf(file_path)
    # Save data in the user session
    cl.user_session.set("docs", docs)
    docsearch = Milvus.from_documents(documents = docs, 
                                      embedding = embeddings,
                                      collection_name = "cozmos",
                                      connection_args={"uri": URI})
    return docsearch 

# System template
system_prompt = (
    "You are a friendly assistant for customer support question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "You are not a general purpose chatbot, you’re here for our business needs (KALBE), and your job is on the line!"
    "Do not answer the question if the docs is not relevant with the question."
    "\n\n"
    "{docs}"
)

def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    model = llm
    
    # Retriever check
    if not cl.user_session.get("docsearch"):
        local_pdf_path = "./data/WORKSHEET OSKM ITB 2024.pdf"    
        docsearch = get_docsearch_from_pdf(local_pdf_path)
        cl.user_session.set("docsearch", docsearch)
    else:
        docsearch = cl.user_session.get("docsearch")
    
    retriever = docsearch.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | RunnableLambda(lambda inputs: {
              **inputs,
              "docs": retriever.invoke(inputs["question"])
        })
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


@cl.password_auth_callback
def auth(username: str, password: str):
    if (username, password) == ("admin1", "admin1"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    setup_runnable()
    await cl.Message(content="Selamat datang di Cosmoz Chatbot! Silakan bertanya tentang MOSTRANS 😁").send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    runnable = cl.user_session.get("runnable")  # type: Runnable

    res = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)

    await res.send()

    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)