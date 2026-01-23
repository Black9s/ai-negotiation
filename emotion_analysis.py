import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

class EmotionAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def analyze_text(self, text):
        """
        Analyze the emotion in a given text using multiple methods.
        Returns a dictionary with sentiment scores.
        """
        # VADER sentiment analysis
        vader_scores = self.sia.polarity_scores(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment
        
        # Combine results
        results = {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_sentiment.polarity,
            'textblob_subjectivity': textblob_sentiment.subjectivity
        }
        
        return results
    
    def get_emotion_label(self, compound_score):
        """
        Convert compound score to emotion label
        """
        if compound_score >= 0.5:
            return 'Very Positive'
        elif compound_score > 0:
            return 'Positive'
        elif compound_score == 0:
            return 'Neutral'
        elif compound_score > -0.5:
            return 'Negative'
        else:
            return 'Very Negative'
    
    def analyze_multiple_texts(self, texts):
        """
        Analyze multiple texts and return a DataFrame with results
        """
        results = []
        for text in texts:
            analysis = self.analyze_text(text)
            analysis['text'] = text
            analysis['emotion'] = self.get_emotion_label(analysis['vader_compound'])
            results.append(analysis)
        
        return pd.DataFrame(results)
    
    def plot_emotion_distribution(self, df):
        """
        Create a bar plot of emotion distribution
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='emotion')
        plt.title('Distribution of Emotions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    analyzer = EmotionAnalyzer()
    
    # Example texts
    texts = [
        "I am so happy today! Everything is going great!",
        "I feel a bit sad about the weather.",
        "The movie was okay, nothing special.",
        "I'm extremely angry about the service!",
        "What a wonderful day to be alive!"
    ]
    
    # Analyze texts
    results_df = analyzer.analyze_multiple_texts(texts)
    
    # Print results
    print("\nEmotion Analysis Results:")
    print(results_df[['text', 'emotion', 'vader_compound']])
    
    # Plot distribution
    analyzer.plot_emotion_distribution(results_df)

if __name__ == "__main__":
    main() 