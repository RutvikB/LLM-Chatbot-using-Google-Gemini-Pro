from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

class ChatGemini():
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            stream=True,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            temperature=0,
            max_output_tokens= 100,
            max_retries=5
        )

        self.prompt = PromptTemplate.from_template(
            """You are an AI assistant for question-answering tasks. 
            Answer the input question considering the history of the conversation. 
            If you don't know the answer, just say that you don't know. 
            Use five sentences maximum and keep the answer concise.
            Chat history: {chat_history}
            User question: {user_query}
            Response:"""
        )

    def invoke_model(self, query:str, history):
        """
        Method to return the response using the streaming chain
        :param user_query:
        :param chat_history:
        :return:
        """
        self.chain = self.prompt | self.model | StrOutputParser()

        output_stream = self.chain.stream(
            {
                "chat_history": history,
                "user_query": query
            }
        )

        return output_stream






# chat_model = ChatGoogleGenerativeAI(
#             model="gemini-1.5-pro",
#             stream=True,
#             safety_settings={
#                 HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#             },
#             temperature=0,
#             max_output_tokens= 100,
#             max_retries=5
#         )