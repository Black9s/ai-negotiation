from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_text(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment

# Example usage
user_text = input("Enter negotiation message: ")
analysis = analyze_text(user_text)
print("Text Analysis:", analysis)
