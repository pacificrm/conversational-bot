from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
from chatbot.loader import load_and_split_documents
from chatbot.vectorstore import initialize_pinecone
from chatbot.config import MODEL_NAME

# Setup embeddings
embeddings = HuggingFaceEmbeddings()

# Initialize model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map='auto',
    torch_dtype=torch.float16,
    use_auth_token=True,
    load_in_8bit=True
)

# Setup pipeline and LLM
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0})

# Load and split documents
urls = [
    'https://blog.gopenai.com/paper-review-llama-2-open-foundation-and-fine-tuned-chat-models-23e539522acb',
    'https://www.mosaicml.com/blog/mpt-7b',
    'https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models',
    'https://lmsys.org/blog/2023-03-30-vicuna/'
]
documents = load_and_split_documents(urls)

# Initialize vector store
vectorstore = initialize_pinecone("llama", embeddings, documents)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def process_query(query):
    result = qa_chain.run(query)
    return result
