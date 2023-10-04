from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from datasets import load_from_disk
import os

cuda = torch.cuda.is_available()
print(f'CUDA available? \n{cuda}')

# huggingface hub model id
model_id = "google/flan-t5-large"

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Define LoRA Config
lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)
# prepare int-8 model for training
model = prepare_model_for_int8_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

data_path = os.path.join(os.getcwd(), 'data')
output_dir="lora-flan-t5-large"
train_dataset = load_from_disk(os.path.join(data_path, 'train'))
eval_dataset = load_from_disk(os.path.join(data_path, 'test'))

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
	auto_find_batch_size=True,
    learning_rate=1e-3, # higher learning rate
    num_train_epochs=1,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    report_to="tensorboard",
    load_best_model_at_end=True,
    predict_with_generate=True,
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()

save_id = 'lora-flan-t5-large-uspc'
trainer.model.save_pretrained(save_id)
tokenizer.save_pretrained(save_id)