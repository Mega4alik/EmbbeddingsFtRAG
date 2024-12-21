import json
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from safetensors.torch import load_file
import modeling
from modeling import ChunkQuestionTripletDataset, ChunkQuestionTripletTestDataset, DataCollatorForTripletLoss, ChunkQuestionTargetDataset, ChunkQuestionTargetTestDataset, DataCollatorForCosineEmbeddingLoss, SiameseModel, JinaModel, JinaModel2
from utils import file_get_contents, cosine_similarity, myOpenAI


def prepare_dataset(): # chunks:[chunk1, chunk2, ...],  questions:[ [Q1_1, Q1_2,...], [Q2_1, Q2_2,...], ... ]
    chunks, questions = [], []
    d = json.loads(file_get_contents("./data/chunks3.json"))
    for key in d:
        a = d[key]
        for chunk in a:
            text = "#"+chunk["keywords"]+"\n"+chunk["content"]
            chunks.append(text)
            questions.append( ["Q: "+question for question in chunk["questions_list"]] )
    return (chunks, questions)


class MyTrainer(Trainer):
    @torch.inference_mode()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        outputs = []
        for step, inputs in enumerate(eval_dataloader):
            out = self.model.forward(**inputs) #**input_ids=<>, attention_mast=<>
            x = out["loss"].item()            
            outputs.append(x)
        average = sum(outputs) / len(outputs)
        r = {"eval_loss":average}
        print("evaluate:", r)
        return r
    

def get_embedding(text, task='retrieval.query'):
    embedding = siamese_model.generate([text], task)[0]
    #embedding = opai.get_embedding(text) #openai embedding for comparison
    return embedding


def test():
    chunks, questions = prepare_dataset()
    dataset = ChunkQuestionTripletDataset(chunks, questions, tokenizer)

    n, chunks_emb, yes_count = len(chunks), [], 0
    for i in range(n): chunks_emb.append( get_embedding(chunks[i], task='retrieval.passage') )

    for i in range(n):
        _, question, _  = dataset.test_samples[i]
        idx = i
        qe, a = get_embedding(question), []
        for j in range(n): a.append((j, cosine_similarity(qe, chunks_emb[j])))
        a = sorted(a, key=lambda x: x[1], reverse=True)
        top = [x[0] for x in a[:10]] #top5
        if idx in top: yes_count+=1        
        print(idx, "--", top, ("YES" if idx in top else "NO") )        
    print(f"YES: {yes_count}, NO: {n-yes_count}")


# =================================================

train_mode = 1 #1-train, 2-evaluate
model_type = 1 #1-jinaai, 2-gte-Qwen2
training_type = 2 #1-TripletMarginLoss, 2-CosineEmbeddingLoss

# =================================================

model_name = "jinaai/jina-embeddings-v3" if model_type==1 else "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
modeling.tokenizer = tokenizer

if model_type==1 and training_type==1: siamese_model = JinaModel(model_name=model_name)
elif model_type==1 and training_type==2: siamese_model = JinaModel2(model_name=model_name)
else: siamese_model = SiameseModel(model_name=model_name)

if train_mode==2: #=== Evaluate trained model ===
    modeling.device = torch.device("cuda:0")
    #load trained model weights
    state_dict = load_file("./model_temp/checkpoint-6200/model.safetensors")
    siamese_model.load_state_dict(state_dict)
    siamese_model.cuda()
    siamese_model.eval()
    test()
elif train_mode==1: #=== Train ===
    # Create the dataset
    chunks, questions = prepare_dataset()
    if training_type==1:
        train_dataset = ChunkQuestionTripletDataset(chunks, questions)
        test_dataset = ChunkQuestionTripletTestDataset(train_dataset.test_samples)
    else:
        train_dataset = ChunkQuestionTargetDataset(chunks, questions)
        test_dataset = ChunkQuestionTargetTestDataset(train_dataset.test_samples)
    print("train, test len:", len(train_dataset.samples), len(train_dataset.test_samples))
    #for x in train_dataset.samples[:10]: print(x[0][:10], x[1][:10], x[2])

    #define loss function and data collator
    modeling.loss_fn = nn.TripletMarginLoss(margin=1.0, p=2) if training_type==1 else nn.CosineEmbeddingLoss()
    data_collator = DataCollatorForTripletLoss(tokenizer=tokenizer) if training_type==1 else DataCollatorForCosineEmbeddingLoss(tokenizer=tokenizer)

    # Start training    
    training_args = TrainingArguments(
        output_dir='./model_temp',
        num_train_epochs=30,
        per_device_train_batch_size=1, #16,
        gradient_accumulation_steps=8,
        #gradient_checkpointing=True, - slows down the training
        learning_rate=1e-6,
        logging_steps=20,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_eval_batch_size=1,
        metric_for_best_model='eval_loss',
        remove_unused_columns=False,
        logging_dir="./logs/",
        report_to="tensorboard",
        #weight_decay=0.01,
    )

    trainer = MyTrainer(
        model=siamese_model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset    
    )

    trainer.train()
    