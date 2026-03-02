from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import asyncio
import os

import base64
import os
from openai import OpenAI


def main():
    file_name = "testdescription2.txt"
    test_description = read_from_file(file_name)
    asyncio.run(chat_with_agent(test_description))
    ai_analysis()


def read_from_file(file_name):
    try:
        with open(file_name, "r") as file:
            content = file.read()
            print(content)
            return content
    except FileNotFoundError:
        print("Error: The file 'my_file.txt' was not found.")





async def chat_with_agent(test_description):
    load_dotenv()
    #model = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")
    model = ChatAnthropic(model_name="claude-haiku-4-5-20251001")

    #llm = ChatGoogleGenerativeAI(model="gemini-pro")

    # tools used is playwright data mcp, specify the commands to load mcp client within python script with our agent, similar to what was done in the claude desktop app
    server_params = StdioServerParameters(
        command="npx",
        env={
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
        },
        # make sure to update the full absolute path to your math_server.py file
        args=["@playwright/mcp@latest"],
    )
    async with stdio_client(server_params) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model,tools)


            #start conversation history
            #encourages agent to use multiple of the tools
            messages = [
                {
                    "role": "system",
                    "content": "Think step by step web automation.  Move through the steps faster with less wait time. Less rendering wait time."
                }
            ]


            user_input= "Testing Description: " + test_description

            #add user history (adding more context)
            messages.append({"role": "user", "content": user_input})

            #call the agent
            agent_response = await agent.ainvoke({"messages": messages})

            #extract agents reply and add to history
            ai_message = agent_response["messages"][-1].content
            print(f"Agent: {ai_message}")
            return ai_message

def ai_analysis():
    print("******Performing OCR Analysis of Betting ODDS*****")
    load_dotenv()
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Function to encode the image to base64

    with open("oddsimage.png", "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Prepare the payload for the API call
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use GPT-4o or another vision-enabled model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the over under any odds data text using OCR from this image and return formatted json."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()