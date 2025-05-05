import os
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI  # Make sure you're importing from openai directly
import gradio as gr
import base64
from io import BytesIO
from PIL import Image

load_dotenv(override=True)

system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."

azure_openai = AzureChatOpenAI(
        temperature=0,
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_api_version="2024-12-01-preview",  # Specify API version
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        azure_deployment="gpt-4o",
        model="gpt-4o"
    )

# Initialize Azure OpenAI client for image generation
azure_image_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY_DALL_E"],
    api_version="2024-02-01",
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    azure_deployment="dall-e-3" 
)


# gr.ChatInterface(fn=chat, type="messages").launch()

# Let's start by making a useful function

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")

# There's a particular dictionary structure that's required to describe our function:

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": price_function}]


def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city,"price": price}),
        "tool_call_id": tool_call.id
    }
    return response, city

# gr.ChatInterface(fn=chat, type="messages").launch()

def artist(city):
    try:
        import requests
        
        headers = {
            "Content-Type": "application/json",
            "api-key": os.environ["AZURE_OPENAI_API_KEY_DALL_E"]  # Azure uses api-key header
        }
        
        data = {
            "model": "dall-e-3",
            "prompt": f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
            "size": "1024x1024",
            "style": "vivid", 
            "quality": "standard",
            "n": 1,
            "response_format": "b64_json"
        }
        
        response = requests.post(
            "https://open-ai-key-align.openai.azure.com/openai/deployments/dall-e-3/images/generations?api-version=2024-02-01",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            return None
            
        response_data = response.json()
        image_base64 = response_data["data"][0]["b64_json"]
        image_data = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_data))
    except Exception as e:
        print(f"Failed to generate image for {city}: {str(e)}")
        return None

def chat(history):
    messages = [{"role": "system", "content": system_message}] + history
    response = azure_openai.invoke(messages, tools=tools)
    
    image = None
    
    # LangChain's response is an AIMessage object
     # LangChain's AIMessage has tool_calls in additional_kwargs
    if hasattr(response, "additional_kwargs") and response.additional_kwargs.get("tool_calls"):
        # Extract the tool call information from additional_kwargs
        tool_call = response.additional_kwargs["tool_calls"][0]
        
        # Dictionary access (it's always a dict in additional_kwargs)
        arguments = json.loads(tool_call["function"]["arguments"])
        tool_call_id = tool_call["id"]
        function_name = tool_call["function"]["name"]
            
        city = arguments.get('destination_city')
        price = get_ticket_price(city)
        
        # Create a tool response
        tool_response = {
            "role": "tool",
            "content": json.dumps({"destination_city": city, "price": price}),
            "tool_call_id": tool_call_id
        }
        
        # Add messages to the conversation history
        lc_messages = messages.copy()
        
        # Create tool_call_dict in the correct format
        tool_call_dict = {
            "id": tool_call_id,
            "function": {
                "name": function_name,
                "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments
            },
            "type": "function"
        }
        
        lc_messages.append({"role": "assistant", "content": response.content, "tool_calls": [tool_call_dict]})
        lc_messages.append(tool_response)
        
        # Generate image
        try:
            image = artist(city)
        except Exception as e:
            print(f"Error generating image: {e}")
            # Continue without image if there's an error
        
        # Get final response
        final_response = azure_openai.invoke(lc_messages)
        reply = final_response.content
    else:
        # No tool calls, just use the response content
        reply = response.content
    
    history += [{"role": "assistant", "content": reply}]
    
    return history, image

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row():
        clear = gr.Button("Clear")

    def do_entry(message, history):
        history += [{"role":"user", "content":message}]
        return "", history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot, image_output]
    )
    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

ui.launch(inbrowser=True)