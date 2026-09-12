from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

result = embedding_model.embed_query("Python is used in AI")

print("Embedding generated!")
print("Dimensions:", len(result))
