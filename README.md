# PRODIGY_DS_04 – Social Media Sentiment Analysis

## Project Overview
This project analyzes and visualizes sentiment patterns in social media data to understand public opinions and attitudes toward different brands and topics.

The project uses Python libraries to clean Twitter data, analyze sentiment categories, and generate visualizations that help identify public reactions.

## Objective
To perform sentiment analysis on social media data and understand whether public opinions are Positive, Negative, Neutral, or Irrelevant.

## Dataset
- **Dataset:** Twitter Training Dataset
- **File:** `twitter_training.csv`
- **Source:** Prodigy InfoTech Data Science Internship Dataset

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- WordCloud
- Regular Expressions (re)

## Data Preprocessing
- Loaded the Twitter dataset.
- Handled missing values.
- Removed duplicate records.
- Standardized sentiment labels.
- Removed URLs, mentions, hashtags, special characters, and unnecessary spaces from tweets.
- Created a cleaned text column for analysis.

## Sentiment Categories
- **Positive:** Tweets expressing favorable opinions.
- **Negative:** Tweets expressing unfavorable opinions.
- **Neutral:** Tweets expressing neither positive nor negative opinions.
- **Irrelevant:** Tweets unrelated to the specified entity or topic.

## Visualizations
The project generates the following visualizations:

1. Sentiment Distribution
2. Top 10 Most Discussed Brands and Topics
3. Sentiment Comparison Across Top 10 Entities
4. Positive Sentiment Word Cloud
5. Negative Sentiment Word Cloud

## Project Structure
```text
PRODIGY_DS_TASK_04/
│
├── twitter_training.csv
├── task4.py
├── README.md
│
└── outputs/
    ├── cleaned_twitter_data.csv
    ├── sentiment_distribution.png
    ├── top_entities.png
    ├── entity_sentiment.png
    ├── positive_wordcloud.png
    ├── negative_wordcloud.png
    ├── sentiment_percentages.csv
    └── entity_sentiment_counts.csv
```

## How to Run the Project

1. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn wordcloud
```

2. Run the Python script:

```bash
python task4.py
```

3. View the generated charts and cleaned data in the `outputs` folder.

## Conclusion
This project demonstrates how social media data can be analyzed to understand public opinion. Sentiment distribution, entity comparisons, and word clouds help identify positive and negative discussions about different brands and topics.

## Internship
**Organization:** Prodigy InfoTech  
**Internship:** Data Science Internship  
**Task:** Task 04 – Social Media Sentiment Analysis