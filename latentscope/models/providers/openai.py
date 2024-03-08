import os
import time
import tiktoken
from openai import AzureOpenAI
from .base import EmbedModelProvider, ChatModelProvider
from openai import OpenAI
from latentscope.util import get_key

class OpenAIEmbedProvider(EmbedModelProvider):
    def load_model(self):
        api_key = get_key("OPENAI_API_KEY")
        deployment = get_key("AZURE_OPENAI_DEPLOYMENT")
        if api_key is None:
            print("ERROR: No API key found for OpenAI")
            print("Missing 'OPENAI_API_KEY' variable in:", f"{os.getcwd()}/.env")
        os.environ['AZURE_OPENAI_API'] = api_key
        endpoint = get_key("OPENAI_ENDPOINT")
        if endpoint is None:
            endpoint = "https://api.openai.com"
        api_version = get_key("OPENAI_API_VERSION")
        if api_version is None:
            api_version = "2024-03-01-preview"
        if deployment is not None:
            self.client = AzureOpenAI(api_version=api_version,
                                      azure_endpoint=endpoint,
                                      azure_deployment=deployment)
        else:
            self.client = OpenAI(api_key=api_key)
        # special case for the new embedding models
        if self.name in ["text-embedding-3-small", "text-embedding-3-large"]:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoder = tiktoken.encoding_for_model(self.name)

    def embed(self, inputs, dimensions=None):
        time.sleep(0.01) # TODO proper rate limiting
        enc = self.encoder
        max_tokens = self.params["max_tokens"]
        inputs = [b.replace("\n", " ") for b in inputs]
        inputs = [enc.decode(enc.encode(b)[:max_tokens]) if len(enc.encode(b)) > max_tokens else b for b in inputs]
        if dimensions is not None and dimensions > 0:
            response = self.client.embeddings.create(
                input=inputs,
                model=self.name,
                dimensions=dimensions
            )
        else:
            response = self.client.embeddings.create(
                input=inputs,
                model=self.name
            )
        embeddings = [embedding.embedding for embedding in response.data]
        return embeddings


class OpenAIChatProvider(ChatModelProvider):
    def load_model(self):
        api_key = get_key("OPENAI_API_KEY")
        deployment = get_key("AZURE_OPENAI_DEPLOYMENT")
        if api_key is None:
            print("ERROR: No API key found for OpenAI")
            print("Missing 'OPENAI_API_KEY' variable in:", f"{os.getcwd()}/.env")
        if deployment is not None:
            self.client = AzureOpenAI(api_key=api_key,
                                      azure_deployment=deployment)
        else:
            self.client = OpenAI(api_key=api_key)
        self.encoder = tiktoken.encoding_for_model(self.name)

    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.name,
            messages=messages
        )
        return response.choices[0].message.content
