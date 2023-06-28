import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the model and tokenizer
model_name = "import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the model and tokenizer
model_name = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


def generate_text(prompt, max_length=200, top_k=10, num_return_sequences=1):
    sequences = text_generator(
        prompt,
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
    )
    return sequences[0]["generated_text"]


# Local app
prompt = input("Enter the prompt: ")
generated_text = generate_text(prompt)

print("\nGenerated Text:")
print(generated_text)
"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


def generate_text(prompt, max_length=200, top_k=10, num_return_sequences=1):
    sequences = text_generator(
        prompt,
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
    )
    return sequences[0]["generated_text"]


# Local app
prompt = input("Enter the prompt: ")
generated_text = generate_text(prompt)

print("\nGenerated Text:")
print(generated_text)
