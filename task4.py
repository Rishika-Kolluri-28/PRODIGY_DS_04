
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from wordcloud import WordCloud, STOPWORDS

# Set visualization style
sns.set_theme(style="whitegrid")

# Create output folder
os.makedirs("outputs", exist_ok=True)

print("Loading Twitter Dataset...")

# 1. Load Dataset
df = pd.read_csv("twitter_training.csv", header=None)

# Assign column names
df.columns = ["Tweet_ID", "Entity", "Sentiment", "Tweet"]

print("\nDataset Loaded Successfully!")

# 2. Explore Dataset
print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())

# 3. Data Cleaning

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Remove rows with missing important values
df.dropna(subset=["Entity", "Sentiment", "Tweet"], inplace=True)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Clean sentiment labels
df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.title()

# Keep valid sentiment categories
valid_sentiments = ["Positive", "Negative", "Neutral", "Irrelevant"]
df = df[df["Sentiment"].isin(valid_sentiments)]

# 4. Clean Tweet Text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["Clean_Tweet"] = df["Tweet"].apply(clean_text)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)

# Save cleaned dataset
df.to_csv("outputs/cleaned_twitter_data.csv", index=False)

# 5. Sentiment Distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Sentiment",
    order=["Positive", "Negative", "Neutral", "Irrelevant"]
)

plt.title("Social Media Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig("outputs/sentiment_distribution.png", dpi=300)
plt.show()

# 6. Top 10 Most Discussed Entities
top_entities = df["Entity"].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_entities.values,
    y=top_entities.index
)

plt.title("Top 10 Most Discussed Brands and Topics")
plt.xlabel("Number of Tweets")
plt.ylabel("Entity")
plt.tight_layout()
plt.savefig("outputs/top_entities.png", dpi=300)
plt.show()

# 7. Sentiment Comparison by Entity
top_entity_names = top_entities.index

entity_data = df[df["Entity"].isin(top_entity_names)]

sentiment_table = pd.crosstab(
    entity_data["Entity"],
    entity_data["Sentiment"]
)

sentiment_table = sentiment_table.reindex(
    columns=["Positive", "Negative", "Neutral", "Irrelevant"],
    fill_value=0
)

sentiment_table.plot(
    kind="bar",
    figsize=(13, 7)
)

plt.title("Sentiment Comparison Across Top 10 Entities")
plt.xlabel("Entity")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig("outputs/entity_sentiment.png", dpi=300)
plt.show()

# 8. Positive Word Cloud
positive_text = " ".join(
    df[df["Sentiment"] == "Positive"]["Clean_Tweet"]
)

if positive_text.strip():
    positive_wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        stopwords=STOPWORDS
    ).generate(positive_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(positive_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Positive Sentiment Word Cloud")
    plt.tight_layout()
    plt.savefig("outputs/positive_wordcloud.png", dpi=300)
    plt.show()

# 9. Negative Word Cloud
negative_text = " ".join(
    df[df["Sentiment"] == "Negative"]["Clean_Tweet"]
)

if negative_text.strip():
    negative_wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        stopwords=STOPWORDS
    ).generate(negative_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(negative_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Negative Sentiment Word Cloud")
    plt.tight_layout()
    plt.savefig("outputs/negative_wordcloud.png", dpi=300)
    plt.show()

# 10. Calculate Sentiment Percentages
print("\n========== SENTIMENT ANALYSIS ==========")

sentiment_counts = df["Sentiment"].value_counts()

sentiment_percentages = (
    df["Sentiment"].value_counts(normalize=True) * 100
).round(2)

print("\nSentiment Counts:")
print(sentiment_counts)

print("\nSentiment Percentages:")
print(sentiment_percentages)

# 11. Most Discussed Entities
print("\nTop 10 Most Discussed Entities:")
print(top_entities)

# 12. Dominant Sentiment by Entity
print("\nDominant Sentiment for Top Entities:")

dominant_sentiment = sentiment_table.idxmax(axis=1)
print(dominant_sentiment)

# Save sentiment analysis results
sentiment_percentages.to_csv(
    "outputs/sentiment_percentages.csv",
    header=["Percentage"]
)

sentiment_table.to_csv(
    "outputs/entity_sentiment_counts.csv"
)

# 13. Final Summary
print("\n======================================")
print("PRODIGY TASK 04 COMPLETED!")
print("Sentiment Analysis Successful!")
print("Visualizations Saved in outputs Folder")
print("======================================")
