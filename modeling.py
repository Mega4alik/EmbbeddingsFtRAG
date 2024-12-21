import random
#from typing import Any, Dict, List, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

#tokenizer, loss_fn and device static variables are passed from train.py

#=======================================================
class ChunkQuestionTripletDataset(Dataset):
    def __init__(self, chunks, questions, max_length=4000):        
        self.max_length = max_length
        self.samples = []
        self.test_samples = []
        num_chunks = len(chunks)
        for idx, chunk in enumerate(chunks):
            anchor = chunk
            positive_questions = questions[idx]                                
            self.test_samples.append((anchor, positive_questions[0], questions[self.randint_excluding_idx(num_chunks-1, idx)][0])) #each first positive_question goes to test set

            for positive in positive_questions[1:]:
                negative_indices = list(range(num_chunks))
                negative_indices.remove(idx)
                for _ in range(5): #for each positive, pair 3 negatives
                    negative_idx = random.choice(negative_indices)
                    negative_indices.remove(negative_idx)
                    negative_questions = questions[negative_idx]
                    negative = random.choice(negative_questions)
                    self.samples.append((anchor, positive, negative))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        anchor_text, positive_text, negative_text = self.samples[idx]        
        x =  {
            'anchor_text': anchor_text,
            'positive_text': positive_text,
            'negative_text': negative_text,
        }        
        return x

    def randint_excluding_idx(self, n, idx):        
        rand = random.randint(0, n - 1)
        return rand if rand < idx else rand + 1


class ChunkQuestionTripletTestDataset(ChunkQuestionTripletDataset):
    def __init__(self, test_samples):
        self.samples = test_samples


class DataCollatorForTripletLoss:
    def __init__(self, tokenizer, max_length=4000):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        #print("features:", features)
        anchor_texts = [f['anchor_text'] for f in features]
        positive_texts = [f['positive_text'] for f in features]
        negative_texts = [f['negative_text'] for f in features]
        
        # Tokenize and pad the sequences
        anchor_encodings = self.tokenizer(
            anchor_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        positive_encodings = self.tokenizer(
            positive_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        negative_encodings = self.tokenizer(
            negative_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        batch = {
            'input_ids_anchor': anchor_encodings['input_ids'],
            'attention_mask_anchor': anchor_encodings['attention_mask'],
            'input_ids_positive': positive_encodings['input_ids'],
            'attention_mask_positive': positive_encodings['attention_mask'],
            'input_ids_negative': negative_encodings['input_ids'],
            'attention_mask_negative': negative_encodings['attention_mask'],
        }        
        return batch

#=======================================================

class ChunkQuestionTargetDataset(Dataset):
    def __init__(self, chunks, questions, max_length=4000):        
        self.max_length = max_length
        self.samples = []
        self.test_samples = []
        num_chunks = len(chunks)
        for idx, chunk in enumerate(chunks):
            anchor = chunk
            positive_questions = questions[idx]
            self.test_samples.append((anchor, positive_questions[0], 1)) #each first positive_question goes to test set

            for positive in positive_questions[1:]:
                self.samples.append((anchor, positive, 1))

            for negative_question in self.get_random_samples(idx, questions, 5):
                self.samples.append((anchor, negative_question, -1))

            for negative_chunk in self.get_random_samples(idx, chunks, 5):
                self.samples.append((anchor, negative_chunk, -1))


    def get_random_samples(self, cidx, a, k):
        indices, samples = list( range(len(a)) ), []
        indices.remove(cidx)
        for _ in range(k):
            idx = random.choice(indices)
            indices.remove(idx)
            samples.append( random.choice(a[idx]) if isinstance(a, list) else a[idx] )
        return samples


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        anchor_text, question, target = self.samples[idx]
        x =  {
            'anchor_text': anchor_text,
            'question': question,
            'target': target,
        }        
        return x



class ChunkQuestionTargetTestDataset(ChunkQuestionTargetDataset):
    def __init__(self, test_samples):
        self.samples = test_samples


class DataCollatorForCosineEmbeddingLoss:
    def __init__(self, tokenizer, max_length=4000):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        #print("features:", features)
        anchor_texts = [f['anchor_text'] for f in features]
        questions = [f['question'] for f in features]
        targets = [f['target'] for f in features]
        
        # Tokenize and pad the sequences
        anchor_encodings = self.tokenizer(
            anchor_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        question_encodings = self.tokenizer(questions,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        targets = torch.tensor(targets)
        
        batch = {
            'input_ids_anchor': anchor_encodings['input_ids'],
            'attention_mask_anchor': anchor_encodings['attention_mask'],
            'input_ids_question': question_encodings['input_ids'],
            'attention_mask_question': question_encodings['attention_mask'],
            'targets': targets
        }        
        return batch

#=======================================================

class SiameseModel(nn.Module):
    #fine-tuned model - "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    #Used with TripletMarginLoss function where input = (anchor, positive_question, negative_question)

    def __init__(self, model_name=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, token_embeddings, attention_mask):
        # Mean Pooling - Take attention mask into account
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def forward(self, **inputs):
        # Encode anchor
        outputs_anchor = self.encoder(input_ids=inputs['input_ids_anchor'], attention_mask=inputs['attention_mask_anchor'])
        anchor_embedding = F.normalize(self.mean_pooling(outputs_anchor.last_hidden_state, inputs['attention_mask_anchor']))        
        
        # Encode positive
        outputs_positive = self.encoder(input_ids=inputs['input_ids_positive'], attention_mask=inputs['attention_mask_positive'])
        positive_embedding = F.normalize(self.mean_pooling(outputs_positive.last_hidden_state,inputs['attention_mask_positive']))

        # Encode negative
        outputs_negative = self.encoder(input_ids=inputs['input_ids_negative'], attention_mask=inputs['attention_mask_negative'])
        negative_embedding = F.normalize(self.mean_pooling(outputs_negative.last_hidden_state, inputs['attention_mask_negative']))

        loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
        return {'loss': loss}
    
    def generate(self, texts):
        inputs = tokenizer(texts, return_tensors='pt').to(device)
        outputs = self.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        embeddings = F.normalize(self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])).detach().cpu().numpy()
        return embeddings

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)


#=======================================================

class JinaModel(nn.Module):
    #fine-tuned model - "jinaai/jina-embeddings-v3"
    #Used with TripletMargin loss function where input = (anchor, positive_question, negative_question)

    def __init__(self, model_name=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_flash_attn=True)
        for param in self.encoder.roberta.parameters():
            param.requires_grad = True

    def encode(self, input_ids, attention_mask):
        #with torch.enable_grad(): embeddings = self.encoder.encode(texts, task="retrieval.passage", convert_to_tensor=True)
        with torch.enable_grad():
            #encoded_input = self.encoder.roberta.tokenizer(texts, return_tensors="pt") #.to(device)
            token_embs = self.encoder.roberta.forward(input_ids) 
            embeddings = self.encoder.roberta.mean_pooling(token_embs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=0)                
        return embeddings

    def forward(self, **inputs): #modeling_xlm_roberta comment @torch.inference_mode() on encode function
        anchor_embedding = self.encode(inputs['input_ids_anchor'], inputs['attention_mask_anchor']) 
        positive_embedding = self.encode(inputs['input_ids_positive'], inputs['attention_mask_positive'])
        negative_embedding = self.encode(inputs['input_ids_negative'], inputs['attention_mask_negative'])
        loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
        #if self.training:
        return {'loss': loss}
        
    @torch.inference_mode()
    def generate(self, texts, task):
        encoded_input = tokenizer(texts, return_tensors="pt").to(device)
        embeddings = self.encode(encoded_input['input_ids'], encoded_input['attention_mask']).detach().cpu().numpy()
        #embeddings = self.encoder.encode(texts, task=task)
        return embeddings

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)


class JinaModel2(JinaModel):
    #Used with CosinEmbedding loss function where input = (anchor, question, target(1/-1))
        
    def forward(self, **inputs):
        anchor_embedding = self.encode(inputs['input_ids_anchor'], inputs['attention_mask_anchor']) 
        question_embedding = self.encode(inputs['input_ids_question'], inputs['attention_mask_question'])
        target = inputs['targets']
        loss = loss_fn(anchor_embedding, question_embedding, target)
        return {'loss': loss}    

#=======================================================