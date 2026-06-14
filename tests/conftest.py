"""Shared test data used across all metric tests."""

GROUNDED_EXAMPLE = {
    "query": "What is our return policy?",
    "response": "We offer a 30-day full refund at no extra cost. You can return any item within 30 days of purchase.",
    "context": ["All customers are eligible for a 30-day full refund at no extra cost. Items must be returned within 30 days of the original purchase date."],
}

HALLUCINATED_EXAMPLE = {
    "query": "What is our return policy?",
    "response": "We offer a 90-day money-back guarantee with free return shipping worldwide. We also provide a lifetime warranty on all products.",
    "context": ["All customers are eligible for a 30-day full refund at no extra cost."],
}

OFF_TOPIC_EXAMPLE = {
    "query": "What is our return policy?",
    "response": "The weather in San Francisco is typically foggy in the summer months. Temperatures range from 55 to 65 degrees Fahrenheit.",
    "context": ["All customers are eligible for a 30-day full refund at no extra cost."],
}

CONTRADICTORY_EXAMPLE = {
    "query": "Is the product available?",
    "response": "Yes, the product is currently in stock and available for immediate shipping. Unfortunately, the product is out of stock and will not be available until next month.",
    "context": ["Product SKU-1234 is currently in stock."],
}
