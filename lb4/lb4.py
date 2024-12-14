import ollama

generation_model = "llama2"  # for text generation
summarization_model = "mistral"  # for text summarizing


def generate_text(prompt, model):
    response = ollama.generate(model=model, prompt=prompt)
    return response.response.strip()


def summarize_text(text, model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
        {"role": "user", "content": f"Here is a text: '{text}'. Please summarize it in no more than 2-3 sentences."}
    ]
    response = ollama.chat(model=model, messages=messages)
    return response.message.content.strip()


def main():
    generation_prompt = input("Enter your prompt for text generation: ")
    with open("prompt.txt", "w") as prompt_file:
        prompt_file.write(generation_prompt)

    print(f"Generating text using the model {generation_model}...")
    generated_text = generate_text(generation_prompt, generation_model)
    print("\nGenerated text:")
    print(generated_text)

    print(f"\nSummarizing text using the model {summarization_model}...")
    summarized_text = summarize_text(generated_text, summarization_model)
    print("\nSummarized text:")
    print(summarized_text)

    with open("generated_text.txt", "w") as gen_file:
        gen_file.write(generated_text)

    with open("summarized_text.txt", "w") as sum_file:
        sum_file.write(summarized_text)
    print("\nResults saved in 'generated_text.txt' and 'summarized_text.txt'.")


if __name__ == "__main__":
    main()
