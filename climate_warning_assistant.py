import os
import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Cohere
from pydantic import BaseModel

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Initialize Cohere LLM
llm = Cohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-plus",
    temperature=0.5,
    max_tokens=300,
)

# Define state structure
class WeatherState(BaseModel):
    location: str
    weather: dict = None
    summary: str = None
    alert: dict = None

# Step 1: Fetch weather data
def fetch_weather(state: WeatherState):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={state.location}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code != 200 or "weather" not in data:
        raise Exception(f"Failed to fetch weather: {data.get('message', '')}")
    state.weather = data
    return state

# Step 2: Extract weather summary
def extract_summary(state: WeatherState):
    data = state.weather
    description = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    wind = data["wind"]["speed"]
    state.summary = f"{description.capitalize()} with temperature of {temp}°C and wind speed {wind} m/s."
    return state

# Step 3: Prompt template
prompt = PromptTemplate.from_template("""
Generate a safety warning for rural people based on this weather summary.

Location: {location}
Weather Summary: {summary}

Keep it simple, local, and actionable.
""")

# Step 4: Generate warning
generate_alert = (
    RunnableLambda(lambda state: state.model_dump()) |
    prompt |
    llm |
    RunnableLambda(lambda output: {"alert": {"message": output}})
)

# Step 5: Merge alert into state
def merge_alert(state: WeatherState):
    return state  # No change needed, alert is already in state

# Step 6: Workflow
workflow = StateGraph(WeatherState)
workflow.add_node("fetch_weather", RunnableLambda(fetch_weather))
workflow.add_node("extract_summary", RunnableLambda(extract_summary))
workflow.add_node("generate_alert", generate_alert)
workflow.add_node("merge_alert", RunnableLambda(merge_alert))

workflow.set_entry_point("fetch_weather")
workflow.add_edge("fetch_weather", "extract_summary")
workflow.add_edge("extract_summary", "generate_alert")
workflow.add_edge("generate_alert", "merge_alert")
workflow.add_edge("merge_alert", END)

app = workflow.compile()

# Run the assistant
if __name__ == "__main__":
    location = input("📍 Enter location (e.g., Ahmedabad, Surat): ")
    try:
        result = app.invoke({"location": location})
        print("\n⚠️ Climate Warning:")
        print(result["alert"]["message"])
    except Exception as e:
        print("❌ Error:", e)
