import os
import gradio as gr
from pydantic import BaseModel, Field
from agents import Agent, Runner  # Ensure agents.py is in your project folder
from dotenv import load_dotenv

load_dotenv()

# --- STEP 1: DEFINE MODELS FIRST (Required for Agent Initialization) ---
class ClassificationResponse(BaseModel):
    reason: str = Field(..., description="The reasoning behind the classification")
    category: str = Field(..., description="The category of the customer inquiry")
    confidence: float = Field(..., description="The confidence level")

class BookResponse(BaseModel):
    reason: str = Field(..., description="The reason for the book recommendation")
    response: str = Field(..., description="Personalized book recommendation message")

class ClothingResponse(BaseModel):
    reason: str = Field(..., description="The reason for the clothing recommendation")
    response: str = Field(..., description="Personalized clothing recommendation message")

class RetentionResponse(BaseModel):
    offer: str = Field(..., description="The retention offer being provided")
    discount_percentage: float = Field(..., description="Discount percentage offered")
    response: str = Field(..., description="Personalized retention message to the customer")

class OrderResponse(BaseModel):
    human_approval: bool = Field(..., description="Indicates if human approval is required")
    order_item_name: str = Field(..., description="Name of the item to be ordered")
    item_id: int = Field(..., description="Unique identifier of the item")
    order_id: int = Field(..., description="Unique identifier for the order")
    order_amount: float = Field(..., description="Total amount for the order in USD")
    response: str = Field(..., description="Next steps for completing the order")

# --- STEP 2: DEFINE INVENTORY DATA (Required for Agent Instructions) ---
book_inventory = """
Available Books:
1. "The Midnight Library" by Matt Haig - $16.99 (12 in stock)
2. "Atomic Habits" by James Clear - $19.99 (25 in stock)
3. "The Psychology of Money" by Morgan Housel - $18.50 (8 in stock)
4. "Project Hail Mary" by Andy Weir - $21.99 (15 in stock)
5. "The Thursday Murder Club" by Richard Osman - $17.99 (20 in stock)
"""

clothing_inventory = """
Available Women's Clothing:
1. "Elegant Silk Blouse" - Size M - $89.99 (10 in stock)
2. "Classic Denim Jacket" - Size L - $124.99 (5 in stock)
3. "Floral Summer Dress" - Size S - $79.99 (18 in stock)
4. "Cashmere Cardigan" - Size M - $149.99 (7 in stock)
5. "High-Waist Trousers" - Size L - $94.99 (12 in stock)
"""

# --- STEP 3: INITIALIZE AGENTS ---
classification_agent = Agent(
    name="ClassificationAgent",
    model="gpt-4o-mini",
    instructions="Analyze user query and categorize into: books, clothing, retention, or order.",
    output_type=ClassificationResponse
)

book_agent = Agent(
    name="BookAgent",
    model="gpt-4o-mini",
    instructions=f"You are a book specialist. Use this inventory: {book_inventory}",
    output_type=BookResponse
)

clothing_agent = Agent(
    name="ClothingAgent",
    model="gpt-4o-mini",
    instructions=f"You are a clothing specialist. Use this inventory: {clothing_inventory}",
    output_type=ClothingResponse
)

retention_agent = Agent(
    name="RetentionAgent",
    model="gpt-4o-mini",
    instructions="Offer a 25% discount to retain customers. Be persuasive.",
    output_type=RetentionResponse
)

order_agent = Agent(
    name="OrderAgent",
    model="gpt-4o-mini",
    instructions="Guide through placing orders. Always set human_approval=True for finalized orders.",
    output_type=OrderResponse
)

# --- STEP 4: ASYNC WORKFLOW LOGIC ---
async def process_workflow(message, history, state):
    history.append({"role": "user", "content": message})
    
    # 1. Handle Turn-Based Approval
    if state.get("pending_approval"):
        state["pending_approval"] = False
        if message.lower() in ["yes", "y", "approve"]:
            res = "Order approved and sent to shipping! ✅"
        else:
            res = "Order cancelled at your request. ❌"
        history.append({"role": "assistant", "content": res})
        return history, state, "Order", 1.0

    # 2. Run Classification
    results = await Runner.run(classification_agent, history)
    class_out = results.final_output
    
    # 3. Route to Sub-Agent
    agent_map = {
        "books": book_agent,
        "clothing": clothing_agent,
        "retention": retention_agent,
        "order": order_agent
    }
    
    selected_agent = agent_map.get(class_out.category)
    if not selected_agent:
        bot_msg = "I'm sorry, I couldn't categorize your request correctly."
    else:
        agent_results = await Runner.run(selected_agent, history)
        agent_out = agent_results.final_output
        bot_msg = agent_out.response
        
        # 4. Interactive Approval Logic
        if class_out.category == "order" and agent_out.human_approval:
            state["pending_approval"] = True
            bot_msg = f"Order summary: {agent_out.order_item_name} (${agent_out.order_amount:.2f}). Approve this order? (yes/no)"

    history.append({"role": "assistant", "content": bot_msg})
    return history, state, class_out.category, class_out.confidence

# --- STEP 5: GRADIO UI SETUP ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Customer Service Agent")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", label="Chat History", height=450)
            msg = gr.Textbox(placeholder="How can I help you today?", label="Message")
            clear = gr.Button("Clear Conversation")
        with gr.Column(scale=1):
            category_label = gr.Label(label="AI Classification")
            confidence_label = gr.Label(label="Confidence Score")

    # App States
    workflow_state = gr.State({})
    history_state = gr.State([])

    # Event Handlers
    msg.submit(
        process_workflow, 
        [msg, history_state, workflow_state], 
        [chatbot, workflow_state, category_label, confidence_label]
    )
    msg.submit(lambda: "", None, msg)  # Clear text box
    clear.click(lambda: ([], {}, "", 0), None, [chatbot, workflow_state, category_label, confidence_label])

if __name__ == "__main__":
    demo.launch()