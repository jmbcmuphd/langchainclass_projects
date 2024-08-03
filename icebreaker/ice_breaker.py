from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from third_parties.linkedin import scrape_linkedin_profile
import os


if __name__ == "__main__":
    load_dotenv()
    print("Hello LangChain")

    summary_template = """
        given the LinkedIn information {information} about a person from whom I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    # llm = ChatOllama(model="llama3.1")

    chain = summary_prompt_template | llm | StrOutputParser()
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/eden-marco/", mock=True
    )
    # chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    res = chain.invoke(input={"information": linkedin_data})

    print(res)
