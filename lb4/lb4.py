import ollama
import logging
import time

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

generation_model = "llama2"  # for text generation
summarization_model = "mistral"  # for text summarizing
sentiment_model = "mvkvl/sentiments:aya"  # for sentiment analysis


def write_to_file(filename, data):
    with open(filename, "w") as file:
        file.write(data)
    logging.info(f"Data saved to '{filename}'.")


def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    logging.info(f"{func.__name__} successful. Time taken: {end_time - start_time:.2f} seconds.")
    return result


def generate_text(prompt, model):
    logging.info(f"Generating text using the model {model}...")
    response = ollama.generate(model=model, prompt=prompt)
    return response.response.strip()


def summarize_text(text, model):
    logging.info(f"Summarizing text using the model {model}...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
        {"role": "user", "content": f"Here is a text: '{text}'. Please summarize it in no more than 2-3 sentences."}
    ]
    response = ollama.chat(model=model, messages=messages)
    return response.message.content.strip()


def analyze_sentiment_with_ollama(text, model):
    logging.info(f"Analyzing sentiment using the model {model}...")
    response = ollama.generate(model=model, prompt=text, format='json')
    sentiment_data = response.response.strip()
    try:
        sentiment_json = eval(sentiment_data)
        reasoning = sentiment_json.get("reasoning", [])
        sentiment = sentiment_json.get("sentiment", [])
        confidence = sentiment_json.get("confidence", [])
        sentiment_output = ""
        if reasoning:
            sentiment_output += f"Reasoning: {''.join(reasoning)}\n"
        if sentiment:
            sentiment_output += f"Sentiment: {''.join(str(sentiment))}\n"
        if confidence:
            sentiment_output += f"Confidence: {''.join(str(confidence))}\n"
        return sentiment_output.strip() if sentiment_output else "No sentiment data available."

    except Exception as e:
        logging.error(f"Error parsing sentiment data: {e}")
        return "Error in processing sentiment data."


def main():
    logging.info("Program started.")

    generation_prompt = input("Enter your prompt for text generation and analyzing: ")
    write_to_file("prompt.txt", generation_prompt)

    generated_text = measure_time(generate_text, generation_prompt, generation_model)
    write_to_file("generated_text.txt", generated_text)

    sentiment_generated_text = measure_time(analyze_sentiment_with_ollama, generated_text, sentiment_model)
    write_to_file("sentiment.txt", sentiment_generated_text)

    summarized_text = measure_time(summarize_text, generated_text, summarization_model)
    write_to_file("summarized_text.txt", summarized_text)

    logging.info("Results saved successfully.")
    logging.info("Program completed.")


if __name__ == "__main__":
    main()
