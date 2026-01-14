import streamlit as st
import numpy as np


def show():
    """Comparison page for side-by-side XAI visualization"""

    st.markdown('<p class="main-header">XAI Comparison</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Compare multiple explainability outputs side-by-side</p>',
        unsafe_allow_html=True
    )

    # Check if there are results to compare
    if not st.session_state.results_history:
        st.info("No results available yet. Please analyze some files on the Home page first.")
        return

    st.write(f"**Total results available:** {len(st.session_state.results_history)}")

    # Selection section
    st.header("Select Results to Compare")

    # Create a list of result summaries
    result_options = []
    for i, result in enumerate(st.session_state.results_history):
        summary = f"#{i+1}: {result['file_name']} | {result['model']} | {result['xai_method']}"
        result_options.append(summary)

    # Multi-select for comparison
    selected_indices = st.multiselect(
        "Select 2-4 results to compare",
        options=range(len(result_options)),
        format_func=lambda x: result_options[x],
        default=list(range(min(2, len(result_options)))),
        max_selections=4
    )

    if len(selected_indices) < 2:
        st.warning("Please select at least 2 results to compare")
        return

    # Display comparison
    st.header("Comparison View")

    # Create columns based on number of selections
    cols = st.columns(len(selected_indices))

    for idx, col in enumerate(cols):
        result_idx = selected_indices[idx]
        result_data = st.session_state.results_history[result_idx]

        with col:
            st.subheader(f"Result #{result_idx + 1}")

            # Display metadata
            st.caption(f"**File:** {result_data['file_name']}")
            st.caption(f"**Type:** {result_data['file_type']}")
            st.caption(f"**Model:** {result_data['model']}")
            st.caption(f"**XAI:** {result_data['xai_method']}")

            # Display input
            with st.expander("Input", expanded=False):
                st.image(
                    result_data['result']['processed_input'],
                    caption="Input",
                    use_container_width=True
                )

            # Display prediction
            prediction = result_data['result']['prediction'][0]
            prob = float(prediction[0]) if len(prediction.shape) > 0 else float(prediction)

            if result_data['file_type'] == "Audio":
                if prob > 0.5:
                    st.success(f"REAL ({prob*100:.1f}%)")
                else:
                    st.error(f"FAKE ({(1-prob)*100:.1f}%)")
            else:
                if prob > 0.5:
                    st.error(f"MALIGNANT ({prob*100:.1f}%)")
                else:
                    st.success(f"BENIGN ({(1-prob)*100:.1f}%)")

            st.progress(prob)

            # Display explanation
            st.markdown("**XAI Explanation:**")
            if result_data['result']['explanation'] is not None:
                st.image(
                    result_data['result']['explanation'],
                    caption=result_data['xai_method'],
                    use_container_width=True
                )
            else:
                st.warning("No explanation available")

    # Analysis section
    st.header("Comparison Analysis")

    _display_comparison_analysis(selected_indices)

    # Clear history button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Clear All Results History", type="secondary"):
            st.session_state.results_history = []
            st.rerun()


def _display_comparison_analysis(selected_indices):
    """Display analysis of the comparison"""

    results = [st.session_state.results_history[i] for i in selected_indices]

    # Check if comparing same file with different methods
    file_names = [r['file_name'] for r in results]
    models = [r['model'] for r in results]
    xai_methods = [r['xai_method'] for r in results]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Comparison Insights")

        if len(set(file_names)) == 1:
            st.info(f"Comparing different analyses of the **same file**: {file_names[0]}")

            if len(set(xai_methods)) > 1:
                st.write("**Different XAI methods** show how various explainability techniques "
                         "highlight different aspects of the model's decision.")

            if len(set(models)) > 1:
                st.write("**Different models** may focus on different features when making predictions.")

        else:
            st.info(f"Comparing analyses across **{len(set(file_names))} different files**")

    with col2:
        st.subheader("Summary Statistics")

        # Get predictions
        predictions = []
        for r in results:
            pred = r['result']['prediction'][0]
            prob = float(pred[0]) if len(pred.shape) > 0 else float(pred)
            predictions.append(prob)

        avg_confidence = np.mean(predictions)
        std_confidence = np.std(predictions)

        st.metric("Average Confidence", f"{avg_confidence*100:.2f}%")
        st.metric("Std Dev", f"{std_confidence*100:.2f}%")

        if std_confidence > 0.2:
            st.warning("High variance in predictions - models disagree significantly")
        else:
            st.success("Low variance - consistent predictions across analyses")

    # Method comparison table
    with st.expander("Detailed Comparison Table"):
        import pandas as pd

        comparison_data = []
        for idx in selected_indices:
            r = st.session_state.results_history[idx]
            pred = r['result']['prediction'][0]
            prob = float(pred[0]) if len(pred.shape) > 0 else float(pred)

            comparison_data.append({
                'File': r['file_name'],
                'Type': r['file_type'],
                'Model': r['model'],
                'XAI Method': r['xai_method'],
                'Confidence': f"{prob*100:.2f}%",
                'Prediction': 'Positive' if prob > 0.5 else 'Negative'
            })

        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
