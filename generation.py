import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model_name = "gpt2"  # You can also use "gpt2-medium", "gpt2-large", or "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set padding token to EOS token (since GPT-2 doesn't have a dedicated pad token)
tokenizer.pad_token = tokenizer.eos_token

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Function to generate text with improved sampling & beam search
def generate_text(prompt, max_length=150):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)  # Explicit attention mask

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,  
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,  # Avoid repeating phrases
            do_sample=True,  # ✅ Enable sampling for better creativity
            top_k=50,  # ✅ Consider top 50 words
            top_p=0.95,  # ✅ Use nucleus sampling (95% probability)
            temperature=0.7,  # ✅ Add randomness (lower = more predictable)
            num_beams=5,  # ✅ Beam search for better text quality
            early_stopping=True,  # ✅ Stop when a coherent sentence is formed
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Main loop for user input
if __name__ == "__main__":
    print("Welcome to GPT-2 Text Generator!")
    while True:
        user_prompt = input("Enter a prompt (or type 'exit' to quit): ")
        if user_prompt.lower() == "exit":
            print("Goodbye! 👋")
            break

        generated_text = generate_text(user_prompt, max_length=200)
        print("\nGenerated Text:\n", generated_text)
        print("\n" + "="*50 + "\n")
