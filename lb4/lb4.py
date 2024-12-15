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


def generate_text(prompt, model):
    logging.info(f"Generating text using the model {model}...")
    start_time = time.time()
    response = ollama.generate(model=model, prompt=prompt)
    end_time = time.time()
    generation_time = end_time - start_time
    logging.info(f"Text generation successful. Time taken: {generation_time:.2f} seconds.")
    return response.response.strip()


def summarize_text(text, model):
    logging.info(f"Summarizing text using the model {model}...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
        {"role": "user", "content": f"Here is a text: '{text}'. Please summarize it in no more than 2-3 sentences."}
    ]
    start_time = time.time()
    response = ollama.chat(model=model, messages=messages)
    end_time = time.time()
    summarization_time = end_time - start_time
    logging.info(f"Summarization successful. Time taken: {summarization_time:.2f} seconds.")
    return response.message.content.strip()


def analyze_sentiment_with_ollama(text, model):
    logging.info(f"Analyzing sentiment using the model {model}...")
    start_time = time.time()
    response = ollama.generate(model=model, prompt=text, format='json')
    end_time = time.time()
    sentiment_time = end_time - start_time
    logging.info(f"Analyzing sentiment successful. Time taken: {sentiment_time:.2f} seconds.")

    sentiment_data = response.response.strip()
    try:
        sentiment_json = eval(sentiment_data)  # This will convert string to dict
        reasoning = sentiment_json.get("reasoning", [])
        sentiment = sentiment_json.get("sentiment", [])
        confidence = sentiment_json.get("confidence", [])

        sentiment_output = "Reasoning: " + ("".join(reasoning) if reasoning else "[]") + "\n"
        sentiment_output += "Sentiment: " + ("".join(str(sentiment)) if sentiment else "[]") + "\n"
        sentiment_output += "Confidence: " + ("".join(str(confidence)) if confidence else "[]") + "\n"

        return sentiment_output.strip()

    except Exception as e:
        logging.error(f"Error parsing sentiment data: {e}")
        return "Error in processing sentiment data."


def main():
    logging.info("Program started.")

    generation_prompt = input("Enter your prompt for text generation: ")

    with open("prompt.txt", "w") as prompt_file:
        prompt_file.write(generation_prompt)
    logging.info(f"Prompt saved to 'prompt.txt'.")

    generated_text = generate_text(generation_prompt, generation_model)
    logging.info(f"Generated text:\n{generated_text}")

    with open("generated_text.txt", "w") as gen_file:
        gen_file.write(generated_text)
    logging.info(f"Generated text saved to 'generated_text.txt'.")

    sentiment_generated_text = analyze_sentiment_with_ollama(generated_text, sentiment_model)
    logging.info(f"Sentiment of generated text:\n{sentiment_generated_text}")
    with open("sentiment.txt", "w") as gen_file:
        gen_file.write(sentiment_generated_text)

    summarized_text = summarize_text(generated_text, summarization_model)
    logging.info(f"Summarized text:\n{summarized_text}")

    with open("summarized_text.txt", "w") as sum_file:
        sum_file.write(summarized_text)
    logging.info(f"Summarized text saved to 'summarized_text.txt'.")

    logging.info("Results saved successfully.")
    logging.info("Program completed.")


if __name__ == "__main__":
    main()
