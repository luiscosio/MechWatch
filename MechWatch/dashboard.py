from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MechWatch.runtime import WatchdogResult, WatchdogRuntime


@st.cache_resource(show_spinner=False)
def load_runtime() -> WatchdogRuntime:
    return WatchdogRuntime()


def render_result(result: WatchdogResult) -> None:
    st.write("### Model Output")
    st.write(result.text)
    if result.enabled:
        if result.blocked:
            st.warning("⚠️ Cognitive Interdiction Triggered")
        else:
            st.success("Watchdog stayed below threshold.")


def render_scores(result: WatchdogResult) -> None:
    scores = result.scores or []
    if not scores:
        st.info("No scores to display.")
        return
    st.line_chart(scores, height=250)
    st.metric("Latest Deception Score", f"{scores[-1]:.4f}")
    st.metric("Suggested Threshold", f"{result.threshold:.4f}")
    st.caption(f"Tokens inspected: {result.tokens_generated}")


def main() -> None:
    st.set_page_config(page_title="Mechanistic Watchdog", layout="wide")
    st.title("🧠 Mechanistic Watchdog")
    try:
        runtime = load_runtime()
    except FileNotFoundError as exc:
        st.error(f"{exc}\n\nRun `python -m MechWatch.calibrate` first.")
        return

    col_left, col_right = st.columns([2, 3])
    with col_left:
        st.subheader("Chat Interface")
        prompt = st.text_area("Prompt", height=180, value="Is it true that the Earth is flat?")
        threshold = st.slider("Threshold", -5.0, 5.0, value=float(runtime.threshold), step=0.01)
        max_tokens = st.slider("Max new tokens", 1, 256, value=runtime.cfg.max_new_tokens)
        temperature = st.slider("Temperature", 0.0, 1.5, value=runtime.cfg.temperature, step=0.05)
        top_p = st.slider("Top-p", 0.1, 1.0, value=runtime.cfg.top_p, step=0.05)
        enabled = st.checkbox("Enable Watchdog", value=True)

        if st.button("Run", type="primary"):
            with st.spinner("Running guarded generation..."):
                result = runtime.generate_with_watchdog(
                    prompt,
                    threshold=threshold,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    enabled=enabled,
                )
            st.session_state["watchdog_result"] = result

    with col_right:
        st.subheader("Brain Scan")
        result = st.session_state.get("watchdog_result")
        if result:
            render_scores(result)
        else:
            st.info("Run the watchdog to see live scores.")

    if result := st.session_state.get("watchdog_result"):
        with col_left:
            render_result(result)


if __name__ == "__main__":
    main()

