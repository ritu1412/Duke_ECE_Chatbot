from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain import HuggingFacePipeline
from langchain.chains import LLMChain
from dotenv import load_dotenv

import gradio as gr
import chromadb
import torch
import os

load_dotenv()

hf_token = os.getenv("HF_READ_TOKEN")
openai_token = os.getenv("OPENAI_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name='mistralai/Mistral-7B-Instruct-v0.2'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,token = hf_token,trust_remote_code=True)
model.to(device)

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=1000,
    pad_token_id=tokenizer.eos_token_id,
    token = hf_token
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)



def get_conversational_chain(langchain_chroma):
    prompt_template = """
    ### [INST] Instruction: Answer the question based on your knowledge. Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    # Create prompt from prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Create llm chain
    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

    rag_chain = (
    {"context": langchain_chroma.as_retriever(), "question": RunnablePassthrough()}
        | llm_chain
    )
    return rag_chain

# get the vectorstore
client = chromadb.PersistentClient(path="DB/")
embeddings = OpenAIEmbeddings(openai_api_key=openai_token)
langchain_chroma = Chroma(client=client,collection_name='ece_bot',
                    embedding_function=embeddings)
conversational_chain = get_conversational_chain(langchain_chroma)


def ECE_chatbot(question,chain):
  bot_message = "Default message"
  bot_message = chain.invoke(question)['text']
  index = bot_message.find('[/INST]')
  return bot_message[index+9:]

def main():
    demo = gr.Blocks()

    chatbot = gr.Interface(
        fn = ECE_chatbot,
        inputs=["textbox"],
        outputs=["textbox"],
        title="ECEBot"
    )

    with demo:
        gr.TabbedInterface(
            [chatbot],
            ["Chatbot"],
        )

    demo.launch(share=True,debug=True)
    
if __name__ == "__main__":
    main()

