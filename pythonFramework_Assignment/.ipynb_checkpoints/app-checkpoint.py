import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("metadata.csv")
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['year'] = df['publish_time'].dt.year
    df['abstract_word_count'] = df['abstract'].fillna("").apply(lambda x: len(x.split()))
    return df

df = load_data()

# App Title
st.title("CORD-19 Data Explorer")
st.write("An interactive exploration of COVID-19 research papers")

# Sidebar filters
year_range = st.slider("Select publication year range:",
                       int(df['year'].min()), int(df['year'].max()),
                       (2020, 2021))

filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

st.write(f"Showing {len(filtered_df)} papers between {year_range[0]} and {year_range[1]}")

# --- Visualization 1: Publications by Year ---
st.subheader("Publications by Year")
year_counts = filtered_df['year'].value_counts().sort_index()
fig, ax = plt.subplots()
sns.barplot(x=year_counts.index, y=year_counts.values, color="skyblue", ax=ax)
ax.set_title("Publications per Year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Papers")
st.pyplot(fig)

# --- Visualization 2: Top Journals ---
st.subheader("Top Journals")
top_journals = filtered_df['journal'].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(x=top_journals.values, y=top_journals.index, palette="viridis", ax=ax)
ax.set_title("Top Journals")
st.pyplot(fig)

# --- Visualization 3: Word Cloud of Titles ---
st.subheader("Word Cloud of Titles")
text = " ".join(str(title) for title in filtered_df['title'].dropna())
if text.strip():
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("No titles available for word cloud.")

# --- Visualization 4: Source Distribution ---
st.subheader("Distribution by Source")
if "source_x" in filtered_df.columns:
    source_counts = filtered_df['source_x'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=source_counts.values, y=source_counts.index, palette="magma", ax=ax)
    ax.set_title("Top Data Sources")
    st.pyplot(fig)
else:
    st.write("No source_x column available in dataset.")

# Show sample of data
st.subheader("Sample Data")
st.dataframe(filtered_df.head(20))
