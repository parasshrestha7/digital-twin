import os
from openai import OpenAI
import gradio as gr
import uuid
import chromadb
from pprint import pprint
import json
import requests
import random

#---------------------------------------------------
# Setup
#---------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("API Key is missing.")
client = OpenAI()


#---------------------------------------------------
# Document
#---------------------------------------------------

#1
document_overview = """
Overview
Hailing from Nepal, the land of Mount Everest, I bring that same spirit of ambitious ascent to the world of analytics and engineering.\
As a Data Analyst and AI Engineer, I operate across the full data lifecycle: from architecting robust data infrastructure and pipelines,\
to surfacing actionable insights through elegant dashboards and research reports. With deep roots in both healthcare and technology, \
I bring a rare combination of domain expertise and technical depth to every problem I tackle. 
What sets me apart is my ability to think at multiple altitudes simultaneously, zooming out to shape business intelligence strategy,\
then zooming in to interrogate data at the row level. I am fast and iterative by nature, yet meticulous in detail. Whether collaborating\
cross-functionally with stakeholders or diving deep into a SQL query at midnight, I adapt fluidly to whatever the moment demands.

Technical Toolkit
- SQL and Databases
- AI Engineering
- Python and the Data Science stack
- BI and Visualization tools
- Excel

Project Experience
Most recently, I deployed a digital twin from scratch, an AI Engineering project spanning Large Language Models, LLM tool calling, and Retrieval\
Augmented Generation. I did not just learn the concepts; I built the system end to end, and it works.

My project portfolio spans the breadth of modern data work. I have designed and built end-to-end dashboards and reporting systems that empower\
teams to make faster, more confident decisions. I have stood up data infrastructure from scratch, including databases, warehouses, and analytics\
platforms, giving organizations a solid foundation to grow on. My exploratory data analysis work uncovers patterns and anomalies that others miss,\
while my data cleaning and governance efforts ensure that the insights produced are trustworthy and defensible.
Beyond the technical, I have contributed to business intelligence strategy, shaping how organizations think about and use their data at an institutional\
level. I have also produced research and insight reports that translate complex analytical findings into compelling narratives for both technical and\
non-technical audiences. Most recently, my work has extended into AI, building and working with intelligent systems that push the boundaries of what data can do.

What Drives Me
Fundamentally, I am driven by the satisfaction of turning chaos into clarity: taking messy, complex data and transforming it into something that\
actually moves the needle. I am energized by hard problems that resist easy solutions, and by the knowledge that my work creates real-world impact\
in domains like healthcare, where the stakes are high and the data is deeply human.
Continuous learning is not just a habit — it is a core part of my identity. Having taught myself many of my most valuable skills, I approach every\
new technology, framework, or domain with genuine curiosity and confidence. I am a builder at heart: nothing excites me more than creating something\
from scratch and watching it deliver measurable business value.

Communication Style
I communicate like a skilled translator, equally fluent in the language of data and the language of people. Thoughtful and measured, I never speak before\
I have something worth saying, and when I do, I back it up with evidence and structured reasoning. But I am far from dry: I am passionate and energetic,\
capable of making a regression analysis feel like a story worth leaning in for. I have a particular gift for making technical concepts accessible to\
non-technical stakeholders, bridging the gap between the data room and the boardroom with ease.

Working Style
I work at a fast and iterative pace, shipping early and refining constantly. I am detail-oriented, ensuring nothing slips through the cracks, while\
remaining team-first in my approach, always bringing people along on the journey. I thrive across changing contexts and challenges, and I am a\
big-picture thinker who never loses sight of the details.

Interests and Hobbies
Outside of work, my curiosity does not switch off. I enjoy gaming, puzzles, and strategy games, drawn to the challenge of systems thinking in recreational form.\
Fitness and sports keep me grounded and disciplined, while music provides a different kind of rhythm to balance the analytical intensity of my work.
Travel is a significant part of my life. Having lived and worked across multiple countries and cultures, I bring a genuinely global perspective to everything I do.\
In the kitchen, I approach cooking with the same experimental mindset I bring to data: tinkering, iterating, and occasionally producing something remarkable.\
And when I am not building things professionally, I am building things personally, as side projects and tinkering are a constant, a natural extension of a mind\
that is always working on something new.

Signature Traits
I am self-taught in several of my most valuable technical skills, a testament to my discipline and autodidactic drive. Globally experienced, having lived and worked\
across multiple countries and cultures, I bring a rich cross-cultural lens to problems and people. I am extraordinarily consistent and disciplined: I show up every day and I deliver. I am always building, perpetually experimenting and pushing the boundaries of what I know. And I am proudly Nepali, from the country of the world's highest peak, carrying a quiet ambition to match.

"""

#2
document_education = """

Bachelor of Pharmacy | 2010 - 2014
MBA | 2017 - 2019

"""

#3
document_professional_experience = """
Present Experience
DATA ANALYST II | HCA HEALTHCARE (PARALLON) | JUNE 2024 – PRESENT
· Develop sophisticated scripts to automate, reducing total processing time by 75% through the implementation of complex business logic for patient identification and validation.
· Initiate multi-phase automation project for recurring clinical data scripting tasks with a strategic roadmap to achieve near 100% automation and vendor delegation and reducing manual workload for operations team.
· Optimize web scraping algorithms to bypass bot-detection protocols, ensuring uninterrupted data retrieval for ICD-10 codes.
· Re-engineer store procedure cleanup logic and integrate automated Azure Data Factory pipeline alerts to eliminate the risk of accidental deletion in production, strengthening data integrity.
· Facilitate staging and restaging of legacy clinical cases for Operations Support team, ensuring accurate historical data representation for reporting.
· Complete data analytics project life cycle (requirements gathering, architecture, design, implementation, and support).
· Responsible for data mapping exercise for applications, data systems.
· Perform data analysis, using in-depth knowledge of databases, non-structured and healthcare data.
· Responsible for analyzing business requirements, designing, and developing quality and patient data registry applications or repositories.
· Provide advanced analysis and ad hoc operational data quality and data literacy reports as requested by stakeholders, business partners, and leadership.
· Present data formally and informally and facilitate discussion regarding data outputs.
· Create documentation for work products and manage or meet target dates.

Past Experience
DATA ANALYST III | GROUPS360 | JUNE 2022 – JUNE 2024
· Create specifications to bring data into a common structure, mapping data structures to meet established requirements, developing data solutions to support analyses, performing analysis, interpreting results, developing actionable insights, and presenting recommendations to support service delivery. 
· Partner with stakeholders to understand data requirements and develop tools and models such as segmentation, data visualizations, decision aids and business case analysis to support the organization. 
· Provide deep analysis on internal data sets and recommend/build aggregates, dashboards, and scorecards to be used for reporting at detail and management level. 
· Produce and manage the delivery of 25+ data and analytics to internal and external stakeholders and clients. 
· Work on embedding Power BI in company application with Row Level Security
· Use business intelligence, data visualization, query, analytic and statistical software to build solutions, perform analysis and interpret data. 
· Design and deploy custom solutions to support enrollment and other business users for data collection, auditing, dashboards, and metrics/KPIs. 
· Participate in strategic & tactical planning discussions. 

"""

#---------------------------------------------------
# Chunking Function
#---------------------------------------------------
def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    Splits text into overlapping chunks with smart boundary snapping.

    Args:
        text:       The input text to split.
        chunk_size: Maximum characters per chunk.
        overlap:    Number of characters each chunk overlaps with the previous.

    Returns:
        List of text chunks.
    """
    def _find_natural_boundary(text_slice: str, min_pos: int) -> int | None:
        boundaries = ["\n\n", "\n", ". ", "? ", "! ", " "]
        for delimiter in boundaries:
            pos = text_slice.rfind(delimiter)
            if pos != -1 and pos > min_pos:
                return pos + len(delimiter)  # cut AFTER the boundary
        return None

    chunks: list[str] = []
    start = 0
    halfway = chunk_size // 2
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            # Last chunk — take whatever remains
            chunks.append(text[start:])
            break
        # Try to snap back to a natural boundary, but only past the halfway point
        raw_slice = text[start:end]
        snapped_pos = _find_natural_boundary(raw_slice, min_pos=halfway)
        if snapped_pos is not None:
            end = start + snapped_pos
        chunks.append(text[start:end])
        # Next chunk starts `overlap` characters before current end
        start = max(start + 1, end - overlap)
    return chunks

#---------------------------------------------------
# RAG: Chunk, Embed & Store in ChromaDB
#---------------------------------------------------

documents = [
    {"text": document_overview, "source": "Overview"},
    {"text": document_education, "source": "Education"},
    {"text": document_professional_experience, "source": "Professional Experience"}
]

chunks = []
ids = []
metadatas = []

for doc in documents:
    #Prepare the lists
    chunks_ = split_text_into_chunks(doc["text"], chunk_size=1000, overlap=100)
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_ = [{"source": doc["source"], "chunk_index": i} for i in range(len(chunks_))]
    #Add to main lists
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

#Print for logs
print(f"Created {len(chunks)} chunks:\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (ID: {ids[i]}, Source: {metadatas[i]['source']}, Index: {metadatas[i]['chunk_index']}, Length: {len(chunk)}):")
    print(chunk)
    print()

#Generate embeddings for all chunks
response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)

embeddings = [item.embedding for item in response.data]

#Verify embeddings for logs
print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")

# Initialize ChromaDB and Store Vectors
#initialize ChromaDB client (persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db_twin")
#Alternative: initialize ChromaDB client (in-memory storage)
# chroma_client = chromadb.Client()

#Get or Create + Empty the collection before adding new data (for testing purposes)
collection = chroma_client.get_or_create_collection(name="digital_twin")
if collection.get()["ids"]:
    collection.delete(collection.get()["ids"])

#Adding data to ChromaDB
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)

pprint(collection.get())

#---------------------------------------------------
# Tools
#---------------------------------------------------

tools = []

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

#Create send_notification function
def send_notification(message: str):
    if pushover_user is None or pushover_token is None:#Handling of potential missing credentials
        return "Notification failed: Pushover not configured."
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data = payload)
    return f"Notification sent: {message}"

#Describe Pushover as an LLM tool
send_notification_function = {
    "name": "send_notification",
    "description": "Sends a push notification to the real Paras. Use this when: \
        1) Someone wants to get in touch, hire, or collaborate\
        - ask for their name and contact details first, then send notification to Paras with the name and contact details. \
        2) You don't know the answer to a question about Paras - send AUTOMATICALLY without asking, include the question so\
        he can add this info later.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The notification message to send to the user's device" }
        },
        "required": ["message"]
    }
}

#Add Pushover to the list of tools for the LLM
tools.append({"type":"function", "function":send_notification_function})

#Simulates rolling a single six-sided die
def dice_roll():
    result = random.randint(1, 6)
    return result

#Describe function for the LLM
roll_dice_function = {
    "name": "dice_roll",
    "description": "Simulates rolling a single six-sided die and return the result. Use this when the user wants to roll a die for games, decisions, or random number generation.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

#Add function to list of tools of LLM
tools.append({"type":"function", "function":roll_dice_function})

#---------------------------------------------------
# Tool Handler
#---------------------------------------------------
def handle_tool_call(tool_calls):
    tool_results = []

    #loop through the tool calls and and handle each call
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        # print(f"Calling function {function_name}") #For future debugging

        #Route to the appropriate function based on the function_name
        if function_name == "send_notification":
            content = send_notification(args["message"])
        elif function_name == "dice_roll":
            content = f"Rolled: {dice_roll()}"
        # elif function_name == "<insert_function_name_3>":
            # content = insert_function_name_3(args["message"])
        # elif function_name == "<insert_function_name_4>":
            # content = insert_function_name_4(args["message"])
        #.....
        else:
            content = f"Unknown function: {function_name}"

        tool_call_result = {
            "role":"tool",
            "content": content,
            "tool_call_id": tool_call.id
        }
        tool_results.append(tool_call_result)
    
    return tool_results

#---------------------------------------------------
# System Meassage
#---------------------------------------------------

system_message = """ You are a digital twin of Paras Shrestha. When people talk to you, 
you respond as Paras - in first person, using his voice, personality, and knowledge.

Important: do not make things up. If you don't know an answer, say you don't know. 
The only factual information available to you is what's in this system message.
You cannot get any for facts about Paras from the internet or make them up.

IMPORTANT: Whenever you don't know something about Paras,
ALWAYS use the send_notification tool to alert the real Paras - do this AUTOMATICALLY without asking the user.
"""

#---------------------------------------------------
# Main Response Function
#---------------------------------------------------

def respond_ai(message, history):
    #RAG: Embed the query using the same model we used for the chunks to ensure compatibility
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = [message]
    )
    query_embedding = response.data[0].embedding

    #RAG: Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    #RAG: Stitch retrieved chunks together to create context for the response
    context = "\n---\n".join(results["documents"][0])
    
    #Pint logs for debugging
    print("\n==================================\n")
    print(f"User message:\n{message}\n")
    print("***Retrieved Chunks:")
    for chunk, metadata in zip(results["documents"][0], results["metadatas"][0]):
        print("--------------------------------")
        print(f"<<Document {metadata['source']} -- Chunk {metadata['chunk_index']}>>\n{chunk}\n")

    #Update system message with context (for this conversation turn)
    system_message_enhanced = system_message + "\n\nContext:\n" + context
    
    #Build message for this turn
    messages = [{"role":"system", "content": system_message_enhanced}] + history + [{"role":"user", "content": message}]
    
    #Call LLM
    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages = messages,
        tools = tools
    )
    message = response.choices[0].message
    
    #Check if model wants to call a tool
    while message.tool_calls:
        pprint(message.tool_calls)

        tool_result = handle_tool_call(message.tool_calls) # whole list of tool calls on purpose
        messages.append(message)
        messages.extend(tool_result) #Changed from append() to extend() when we switched to multiple tool call handling
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message

    #Note: May be consider adding protection from infinite consecutive tool calling

    return(message.content)

#---------------------------------------------------
# Launch Gradio
#---------------------------------------------------

gr.ChatInterface(
    fn=respond_ai,
    title="Paras' Digital Twin",
    # chatbot=(gr.Chatbot(avatar_image=(None, "Paras.jpeg"))),
    description="Chat with an AI version of Paras Shrestha. Ask about his experience, projects, or just say hi!",
    examples=["What's your background?", "Most recent AI Engineering project", "Tell me about your current role at HCA"]
).launch(server_name="0.0.0.0", server_port=7860)
