from prompted.agents.agent import Agent
from prompted import verbosity
import asyncio

from pydantic import BaseModel


class Response(BaseModel):
    response: str


class Context(BaseModel):
    where_are_we: str = "indoors"


verbosity("debug")


async def main():
    # Create a simple agent
    agent = Agent.create(
        name="TestAgent",
        instructions="You are a helpful assistant.",
        model="openai/gpt-4o-mini",
        output_type=Response,
        context=Context(),
        update_context_before_response=True,
        keep_intermediate_steps=False,
        planning=True,
    )

    # Run the agent with a simple query
    response = await agent.async_run("we are now outdoors")

    # Print the output
    print("Response:", response.output)


if __name__ == "__main__":
    asyncio.run(main())
