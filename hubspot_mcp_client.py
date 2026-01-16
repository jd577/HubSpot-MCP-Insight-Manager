import sys
import io
import asyncio
import json
import re
import os
from typing import Optional
from contextlib import AsyncExitStack
import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class HubSpotMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.available_tools = []

    async def connect_to_server(self, server_script_path: str):
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()
        response = await self.session.list_tools()
        self.available_tools = [tool.name for tool in response.tools]
        print(f"\nConnected to HubSpot Server | Tools: {self.available_tools}")

    async def analyze_intent(self, user_input: str) -> dict:
        if not self.groq_api_key:
            return {"tool": "ask_groq", "params": {"question": user_input}, "confidence": 0.0}

        prompt = f"""
        USER QUERY: "{user_input}"
        TOOLS: get_contacts, create_contact, get_deals, search_contact_by_email, analyze_crm_data, ask_groq
        
        RULES:
        - ADD/SAVE/CREATE -> create_contact (extract firstname, lastname, email)
        - SEE/LIST contacts -> get_contacts
        - SEE/LIST deals -> get_deals
        - SEARCH by email -> search_contact_by_email
        - CRM INSIGHTS -> analyze_crm_data (extract query)
        - GENERAL/OTHER -> ask_groq (extract question)

        RETURN JSON ONLY: {{"tool": "name", "params": {{}}, "confidence": 0.0}}
        """
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a professional tool router. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    return json.loads(json_match.group()) if json_match else {"tool": "ask_groq", "params": {"question": user_input}, "confidence": 0.5}
                return {"tool": "ask_groq", "params": {"question": user_input}, "confidence": 0.5}
        except Exception:
            return {"tool": "ask_groq", "params": {"question": user_input}, "confidence": 0.5}

    async def process_query(self, tool_name: str, params: dict):
        if not self.session:
            return

        print(f"\n[EXECUTION] Tool: {tool_name}")
        print(f"[DATA] Params: {params}")
        
        try:
            result = await self.session.call_tool(tool_name, params)
            print("\n--- RESULT ---")
            print(result.content[0].text)
            print("--------------")
        except Exception as e:
            print(f"Execution Error: {e}")

    async def smart_chat_loop(self):
        print("\nHubSpot MCP Intelligence System")
        print("Ready for natural language queries (type 'quit' to exit)\n")

        while True:
            user_input = input("User: ").strip()
            if user_input.lower() == 'quit':
                break

            if user_input in self.available_tools:
                await self.manual_command(user_input)
            else:
                await self.process_smart_command(user_input)

    async def process_smart_command(self, user_input: str):
        intent = await self.analyze_intent(user_input)
        tool_name = intent["tool"]
        params = intent["params"]
        
        if tool_name == "analyze_crm_data" and "query" not in params:
            params["query"] = user_input
        if tool_name == "ask_groq" and "question" not in params:
            params["question"] = user_input
        if tool_name == "search_contact_by_email" and "email" not in params:
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_input)
            if email_match:
                params["email"] = email_match.group(0)

        if tool_name in self.available_tools:
            await self.process_query(tool_name, params)
        else:
            print(f"Error: Tool '{tool_name}' is not recognized by the server.")

    async def manual_command(self, command: str):
        try:
            if command == "get_contacts":
                limit = input("Limit: ").strip()
                await self.process_query(command, {"limit": int(limit) if limit else 10})
            elif command == "create_contact":
                email = input("Email: ").strip()
                fname = input("First Name: ").strip()
                lname = input("Last Name: ").strip()
                await self.process_query(command, {"email": email, "firstname": fname, "lastname": lname})
            elif command == "get_deals":
                limit = input("Limit: ").strip()
                await self.process_query(command, {"limit": int(limit) if limit else 10})
            elif command == "analyze_crm_data":
                query = input("Analysis Query: ").strip()
                await self.process_query(command, {"query": query})
            elif command == "search_contact_by_email":
                email = input("Search Email: ").strip()
                await self.process_query(command, {"email": email})
            elif command == "ask_groq":
                question = input("Question: ").strip()
                await self.process_query(command, {"question": question})
        except Exception as e:
            print(f"Input Error: {e}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py server.py")
        sys.exit(1)

    client = HubSpotMCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.smart_chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())