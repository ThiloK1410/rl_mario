import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
import altair as alt

# --- Configuration ---
DATA_FILE = "training_log.csv"
REFRESH_RATE_SECONDS = 5  # How often the app should check the file for new data

# Set the page title and a descriptive header.
st.set_page_config(layout="wide", page_title="Live Training Data Visualizer")

st.title("📊 Live Training Data Visualizer")
st.write(
    f"This application automatically visualizes data from the `{DATA_FILE}` file. "
    f"It will refresh every {REFRESH_RATE_SECONDS} seconds to show the latest results from your training script."
)

# --- Sidebar for Controls ---
st.sidebar.header("🚀 Experiment Controls")
st.sidebar.write(f"**File:** `{os.path.abspath(DATA_FILE)}`")

# --- File Status Display ---
try:
    if os.path.exists(DATA_FILE):
        mod_time = os.path.getmtime(DATA_FILE)
        st.sidebar.info(f"**Last Updated:**\n{datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.sidebar.warning("File does not exist yet.")
except Exception as e:
    st.sidebar.error(f"Could not get file status: {e}")

# --- Manual Cache Clear Button ---
if st.sidebar.button("Force Reread File", use_container_width=True):
    st.cache_data.clear()
    st.success("Cache cleared. Forcing a full file reread on the next refresh.")

# Button to reset all data by deleting the CSV file.
if st.sidebar.button("Reset Experiment (Deletes File)", use_container_width=True, type="primary"):
    if os.path.exists(DATA_FILE):
        try:
            st.cache_data.clear()  # Clear cache before deleting
            os.remove(DATA_FILE)
            st.sidebar.success(f"Deleted `{DATA_FILE}`. Ready for a new experiment!")
        except Exception as e:
            st.sidebar.error(f"Error deleting file: {e}")
    else:
        st.sidebar.warning("No data file to delete.")
    st.rerun()


# --- Data Loading Function (with Caching Fix) ---
@st.cache_data
def load_data(file_path, file_modification_time):
    """Loads data from the CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        if 'epoch' in df.columns:
            df = df.set_index('epoch')
        return df
    except Exception as e:
        st.session_state.load_error = e
        return None


# --- Main Content Area for Visualization ---
if not os.path.exists(DATA_FILE):
    st.info(
        f"Waiting for data... Please run your training script to create `{DATA_FILE}`."
    )
else:
    if 'load_error' in st.session_state:
        del st.session_state.load_error

    file_mod_time = os.path.getmtime(DATA_FILE)
    df = load_data(DATA_FILE, file_mod_time)

    if 'load_error' in st.session_state:
        st.error(f"Failed to read or process `{DATA_FILE}`.")
        st.exception(st.session_state.load_error)
        st.warning("Showing last successfully loaded data until the file is fixed.")
    elif df is None or df.empty:
        st.info(f"The data file `{DATA_FILE}` is empty. Waiting for data...")
    else:
        st.header("📈 Live Performance Graphs")
        st.write("A chart is automatically generated for each numeric column.")

        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

        if not numeric_columns:
            st.warning("No numeric data columns found in the file to plot.")
        else:
            for column in numeric_columns:
                st.subheader(f"{column.replace('_', ' ').title()} Over Epochs")

                # THE DEFINITIVE FIX:
                # We bypass st.line_chart and build the chart manually with Altair.
                # This gives us full control and avoids the internal validation
                # path that causes the RecursionError.

                # 1. Reset the index so 'epoch' becomes a regular column for Altair.
                chart_data = df[[column]].reset_index()
                # 2. Rename columns for clarity in the chart encoding.
                chart_data.columns = ['epoch', 'value']

                try:
                    # 3. Build the Altair chart object.
                    chart = (
                        alt.Chart(chart_data)
                        .mark_line(point=True, tooltip=True)  # Add points and tooltips for interactivity
                        .encode(
                            x=alt.X("epoch:Q", title="Epoch"),
                            y=alt.Y("value:Q", title=column.replace('_', ' ').title()),
                        )
                    )
                    # 4. Display the chart using the lower-level st.altair_chart function.
                    st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate chart for column '{column}'.")
                    st.exception(e)

        st.header("📋 Raw Data Log")
        st.dataframe(pd.read_csv(DATA_FILE), use_container_width=True)

# --- Auto-refresh logic ---
time.sleep(REFRESH_RATE_SECONDS)
st.rerun()

