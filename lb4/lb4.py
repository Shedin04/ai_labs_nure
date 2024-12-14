import ollama
import logging

# Set up logging configuration
logging.basicConfig(
    filename='app.log',  # Logs will be saved to this file
    level=logging.INFO,  # Log level can be changed to DEBUG, ERROR, etc.
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

generation_model = "llama2"  # for text generation
summarization_model = "mistral"  # for text summarizing


def generate_text(prompt, model):
    logging.info(f"Generating text using the model {model}...")
    response = ollama.generate(model=model, prompt=prompt)
    logging.info(f"Text generation successful.")
    return response.response.strip()


def summarize_text(text, model):
    logging.info(f"Summarizing text using the model {model}...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
        {"role": "user", "content": f"Here is a text: '{text}'. Please summarize it in no more than 2-3 sentences."}
    ]
    response = ollama.chat(model=model, messages=messages)
    logging.info(f"Summarization successful.")
    return response.message.content.strip()


def main():
    logging.info("Program started.")

    generation_prompt = input("Enter your prompt for text generation: ")

    # Save the prompt to a file
    with open("prompt.txt", "w") as prompt_file:
        prompt_file.write(generation_prompt)
    logging.info(f"Prompt saved to 'prompt.txt'.")

    # Generate text
    generated_text = generate_text(generation_prompt, generation_model)

    logging.info(f"Generated text: {generated_text}")

    # Summarize the generated text
    summarized_text = summarize_text(generated_text, summarization_model)

    logging.info(f"Summarized text: {summarized_text}")

    # Save the generated and summarized texts to files
    with open("generated_text.txt", "w") as gen_file:
        gen_file.write(generated_text)
    logging.info(f"Generated text saved to 'generated_text.txt'.")

    with open("summarized_text.txt", "w") as sum_file:
        sum_file.write(summarized_text)
    logging.info(f"Summarized text saved to 'summarized_text.txt'.")

    logging.info("Results saved successfully.")
    logging.info("Program completed.")


if __name__ == "__main__":
    main()
