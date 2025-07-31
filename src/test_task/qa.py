from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from build_index import CHUNK_SIZE
import torch


def load_qa(persist_dir: str = "chroma_store", top_k: int = 4):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=CHUNK_SIZE,
        temperature=0.1,
        device=0 if torch.cuda.is_available() else -1,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain
