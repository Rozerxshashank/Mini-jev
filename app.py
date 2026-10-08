import streamlit as st
import torch
from transformers import pipeline

st.set_page_config(
    page_title="Mini-Jev: Support Inbox Triage",
    layout="centered"
)


def get_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1


@st.cache_resource
def load_model():
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        device=get_device()
    )


classifier = load_model()

ROUTES = [
    "billing",
    "technical problem",
    "sales question",
    "thank you"
]

URGENCY_LEVELS = [
    "it can wait a week",
    "it should be handled today",
    "it needs action right now"
]


def classify(text: str, candidate_labels: list[str], hypothesis_template: str = "This message is about {}.") -> dict[str, float]:
    result = classifier(
        text,
        candidate_labels=candidate_labels,
        hypothesis_template=hypothesis_template
    )
    return {
        label: round(score, 3)
        for label, score in zip(result["labels"], result["scores"])
    }


def calculate_urgency(text: str) -> float:
    probs = classify(text, URGENCY_LEVELS, hypothesis_template="For support, {}.")
    expected_value = sum(
        (URGENCY_LEVELS.index(level) + 1) * p
        for level, p in probs.items()
    )
    return round(expected_value, 2)


def calculate_anger(text: str) -> float:
    probs = classify(
        text,
        ["angry or frustrated", "calm or happy"],
        hypothesis_template="The customer is {}."
    )
    return probs.get("angry or frustrated", 0.0)


st.title("Mini-Jev: support inbox triage")
st.caption("A deterministic zero-shot triage engine. It only decides, it never writes.")

message = st.text_area(
    "Customer message",
    value="I was charged twice this month and nobody is answering.",
    height=120
)

threshold = st.slider(
    "Auto-route only if at least this sure",
    min_value=0.50,
    max_value=0.99,
    value=0.80,
    step=0.01
)

if st.button("Decide", type="primary"):
    if not message.strip():
        st.warning("Please enter a customer message.")
    else:
        with st.spinner("Evaluating..."):
            route_probs = classify(message.strip(), ROUTES)
            top_route, top_prob = next(iter(route_probs.items()))
            urgency = calculate_urgency(message.strip())
            anger = calculate_anger(message.strip())

        if top_prob >= threshold:
            st.success(f"Auto-route to {top_route} ({top_prob:.0%} sure)")
        else:
            st.warning(f"Send to a human. Best guess: {top_route}, only {top_prob:.0%} sure")

        st.bar_chart(route_probs)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("How angry", f"{anger:.0%}")
        with col2:
            st.metric("How urgent (1 to 3)", urgency)
