import os
import torch
import locale
from getpass import getpass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import GitHubIssuesLoader
from langchain_community.llms import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

locale.getpreferredencoding = lambda: "UTF-8"

GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
ACCESS_TOKEN = getpass(GITHUB_PERSONAL_ACCESS_TOKEN)

# Data
loader = GitHubIssuesLoader(repo="huggingface/peft", access_token=ACCESS_TOKEN, include_prs=False, state="all")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
chunked_docs = splitter.split_documents(docs)

# Embeddings & Retriever
db = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Quantized Model
model_name = "HuggingFaceH4/zephyr-7b-beta"

# bnb_config = BitsAndBytesConfig(
    # load_in_4bit=True, 
    # bnb_4bit_use_double_quant=True, 
    # bnb_4bit_quant_type="nf4", 
    # bnb_4bit_compute_dtype=torch.bfloat16
# )

# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(torch.backends.mps.is_available())

model = AutoModelForCausalLM.from_pretrained(model_name, device_map='mps')

tokenizer = AutoTokenizer.from_pretrained(model_name)

# LLM Chain


text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()
retriever = db.as_retriever()
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain
question = "How do you combine multiple adapters?"
llm_chain.invoke({"context": "", "question": question})

rag_chain.invoke(question)
