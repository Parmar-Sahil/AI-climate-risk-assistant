import streamlit as st
from climate_warning_assistant import app as climate_app  # from your LangGraph code

st.set_page_config(page_title="Climate Warning Assistant", layout="centered")

st.title("🌦️ Climate Warning Assistant")

st.markdown(
    """
    Get real-time weather summaries and safety alerts for any location.
    """
)

# Location input
location = st.text_input("📍 Enter location (e.g., Surat):")

if st.button("Get Warning"):
    if not location:
        st.error("Please enter a valid location.")
    else:
        with st.spinner("Fetching climate data..."):
            try:
                # Invoke LangGraph logic
                result = climate_app.invoke({"location": location})
                summary = result.get("summary", "No summary provided.")
                alert_msg = result.get("alert", {}).get("message", "No alert message.")

                st.success("✅ Weather Summary")
                st.write(summary)

                st.warning(f"⚠️ {alert_msg}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

