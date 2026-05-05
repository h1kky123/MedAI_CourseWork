"""
train_lora_qwen.py
Обучение Qwen2.5-1.5B-Instruct через LoRA на медицинских данных
ИСПРАВЛЕННАЯ ВЕРСИЯ (совместимость с новыми transformers + CPU оптимизация)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

# Конфигурация
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "finetune_dataset.jsonl"
OUTPUT_DIR = "./qwen_medical_lora"

# Проверка CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используем устройство: {device}")

if device == "cpu":
    print("⚠️ ВНИМАНИЕ: Обучение на CPU будет медленным.")
    # Для CPU используем меньший батч, но больше накопления градиентов
    BATCH_SIZE = 1
    GRAD_ACCUMULATION = 8
    EPOCHS = 1 # Начнем с 1 эпохи для проверки, что всё работает
else:
    BATCH_SIZE = 2
    GRAD_ACCUMULATION = 4
    EPOCHS = 3

def main():
    # 1. Загрузка токенизатора и модели
    print("Загрузка модели и токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Если есть GPU, используем float16, иначе float32
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )

    if device == "cpu":
        model.to(device)

    # Настройка токенизера для чатов
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Подготовка LoRA конфигурации
    print("Настройка LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Загрузка и подготовка датасета
    print("Загрузка датасета...")
    dataset = load_dataset("json", data_files=DATASET_PATH)

    def tokenize_function(examples):
        messages = examples["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        tokenized = tokenizer(text, truncation=True, max_length=512, padding="max_length")

        # ВАЖНОЕ ИСПРАВЛЕНИЕ: Создаем колонку labels
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Применяем токенизацию
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["messages"])

    # Разделение на train/eval
    split_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)
    train_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]

    # Устанавливаем формат тензоров PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 4. Настройка тренировки
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch", # <-- ИСПРАВЛЕНО ЗДЕСЬ (было evaluation_strategy)
        fp16=device == "cuda",
        bf16=False,
        report_to="none"
    )

    # 5. Запуск Trainer
    # ИСПРАВЛЕНИЕ: Убран аргумент tokenizer из конструктора Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # tokenizer=tokenizer, <--- УДАЛЕНО
    )

    print("Начало обучения...")
    trainer.train()

    # 6. Сохранение результатов
    print("Сохранение модели...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"✅ Обучение завершено! Модель сохранена в {OUTPUT_DIR}")
    print("Теперь можно использовать адаптеры в API.")

if __name__ == "__main__":
    main()