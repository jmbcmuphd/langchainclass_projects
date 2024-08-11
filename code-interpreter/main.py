from dotenv import load_dotenv
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Start...")

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        verbose=True,
    )

    csv_agent.invoke(
        input={"input": "which season has the most episodes?"}
    )
    csv_agent.invoke(
        input={"input": "in the file episode_info.csv, which writer wrote the most episodes? How many did he write?"}
    )


if __name__ == "__main__":
    main()
