"""Simple interactive bot using a local llm and vectorstore"""
import os
os.environ["LANGCHAIN_HANDLER"] = "langchain"
import torch
import transformers
import safetensors
import inspect
from transformers import LlamaTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores.faiss import FAISS
from safetensors.torch import load_file as safe_load
from quant import make_quant # https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/cuda/quant.py
from embeddings import HuggingFaceEmbeddings
from modelutils import find_layers # https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/cuda/modelutils.py
from gptqloader import GPTQLoader

MODEL = "vicuna-13B-1.1-GPTQ-4bit-128g"
MODEL_FILE = "vicuna-13B-1.1-GPTQ-4bit-128g/vicuna-13B-1.1-GPTQ-4bit-128g.no-act-order.pt"

base_model = GPTQLoader.load_quantized(MODEL, MODEL_FILE)

tokenizer = LlamaTokenizer.from_pretrained(MODEL)

pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=2000,
    temperature=0.0,
    top_p=0.95,
    repetition_penalty=1.2,
    device=0,
)

llm = HuggingFacePipeline(pipeline=pipe)

embeddings = HuggingFaceEmbeddings()

vectorstore = FAISS.load_local('./store/', embeddings)
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(llm, 
                                 chain_type="stuff", 
                                 retriever=retriever,
                                 return_source_documents=False
                                 )

def ask_ai():
    """Main method to talk to the ai"""
    while True:
        question = input("Your question: ")
        with torch.autocast("cuda"):
            answer = qa.run(question)
            print(answer)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    ask_ai()
