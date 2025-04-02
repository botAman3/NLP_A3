## Add imports here
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader , Subset
from collections import Counter
import numpy as np
import math
from tqdm import tqdm
from matplotlib import pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 512

def initialise_projections(d_model, d_k, d_v, num_heads):
    Wq = nn.Linear(d_model, d_k * num_heads)
    Wk = nn.Linear(d_model, d_k * num_heads)
    Wv = nn.Linear(d_model, d_v * num_heads)
    return Wq, Wk, Wv

def pairwise_similarities(Q, K):
    return torch.matmul(Q, K.transpose(-2, -1))

def attention_scaled(scores, d_k):
    return scores / math.sqrt(d_k)

def attention_softmax(scaled_scores):
    return F.softmax(scaled_scores, dim=-1)

def compute_outputs(attention_weights, V):
    return torch.matmul(attention_weights, V)

def make_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask.to(DEVICE)

def apply_causal_mask(scores, mask):
    return scores.masked_fill(mask, float('-inf'))

def split_heads(x, num_heads):
    batch_size, seq_len, d_model = x.size()
    d_k = d_model // num_heads
    return x.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

def merge_heads(x):
    batch_size, num_heads, seq_len, d_k = x.size()
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * d_k)

def self_attention(x, Wq, Wk, Wv, num_heads, mask=None):
    Q = split_heads(Wq(x), num_heads)
    K = split_heads(Wk(x), num_heads)
    V = split_heads(Wv(x), num_heads)
    
    scores = pairwise_similarities(Q, K)
    scaled_scores = attention_scaled(scores, Q.size(-1))
    
    if mask is not None:
        scaled_scores = apply_causal_mask(scaled_scores, mask)
    
    attention_weights = attention_softmax(scaled_scores)
    outputs = compute_outputs(attention_weights, V)
    return merge_heads(outputs)

def split_heads_qkv(Q, K, V, num_heads):
    return split_heads(Q, num_heads), split_heads(K, num_heads), split_heads(V, num_heads)


# class FirstNSampler(Sampler):
#     def __init__(self, mask):
#         self.mask = mask

#     def __iter__(self):
#         return (self.indices[i] for i in torch.nonzero(self.mask))

#     def __len__(self):
#         return len(self.mask)

class TextDataset(Dataset):
        def __init__(self, data, max_len , tokenizer):
            self.data = data
            self.max_len = max_len
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            tokens = self.data[idx]
            padded = pad_to_length(tokens, self.max_len, self.tokenizer)
            input_ids = padded[:-1]
            target_ids = padded[1:]
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)
        


def load_and_preprocess_data():
    with open("shakespear_train.txt", "r") as f:
        lines_train = f.readlines()
    with open("shakespear_dev.txt", "r") as f:
        lines_dev = f.readlines()
    # with open("shakespear_test.txt", "r") as f:
    #     lines_test = f.readlines()

    tokens_train = [line.split() for line in lines_train]

    def flat(tokens):
        return [token for line in tokens for token in line]

    token_counts = Counter(flat(tokens_train))

    vocab_size = 10000
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    most_common = [token for token, _ in token_counts.most_common(vocab_size - len(special_tokens))]
    tokenizer = {token: idx for idx, token in enumerate(special_tokens)}
    for i, token in enumerate(most_common, start=len(special_tokens)):
        tokenizer[token] = i
    tokenizer_inv = {v: k for k, v in tokenizer.items()}

    data_train = []
    for line in lines_train:
        tokens = ['<START>'] + line.split() + ['<END>']
        token_ids = [tokenizer.get(token, tokenizer['<UNK>']) for token in tokens]
        data_train.append(token_ids)
    
    data_val = []
    for line in lines_dev:
        tokens = ['<START>'] + line.split() + ['<END>']
        token_ids = [tokenizer.get(token, tokenizer['<UNK>']) for token in tokens]
        data_val.append(token_ids)

    

    train_dataset = TextDataset(data_train, MAX_LEN , tokenizer)
    mask = torch.tensor([i < 10 for i in range(len(data_val))])
    val_dataset = TextDataset(data_val, MAX_LEN , tokenizer)

    return train_dataset, val_dataset, tokenizer, tokenizer_inv

def pad_to_length(tokens, max_len, tokenizer):
    if len(tokens) >= max_len:
        return tokens[:max_len]
    else:
        return tokens + [tokenizer['<PAD>']] * (max_len - len(tokens))

def tokenize(sentence, pad_to_len=None, tokenizer=None, include_stop=True):
    tokens = sentence.split()
    token_ids = [tokenizer['<START>']]
    for token in tokens:
        token_ids.append(tokenizer.get(token, tokenizer['<UNK>']))
    if include_stop:
        token_ids.append(tokenizer['<END>'])
    if pad_to_len is not None:
        token_ids = pad_to_length(token_ids, pad_to_len, tokenizer)
    return token_ids

def decode(tokens, tokenizer_inv, end_at_stop=True, omit_pad=True):
    decoded = []
    for tok in tokens:
        if omit_pad and tok == tokenizer_inv['<PAD>']:
            continue
        decoded.append(tokenizer_inv.get(tok, '<UNK>'))
        if end_at_stop and tok == tokenizer_inv['<END>']:
            break
    return ' '.join(decoded)


def decode(tokens, tokenizer , tokenizer_inv, end_at_stop=True, omit_pad=True):
    """
    Decode tokens to text.
    """
    decoded = []
    for tok in tokens:
        # Get PAD token ID using direct tokenizer (not inverse)
        if omit_pad and tok == tokenizer['<PAD>']:  # Changed tokenizer_inv to tokenizer
            continue
        decoded.append(tokenizer_inv.get(tok, '<UNK>'))
        # Get END token ID using direct tokenizer
        if end_at_stop and tok == tokenizer['<END>']:  # Changed tokenizer_inv to tokenizer
            break
    return ' '.join(decoded)


@torch.no_grad()
def evaluate_losses(data, model, tokenizer, bs=32, progress=True, pad_to_len=MAX_LEN):
    it = range(0, len(data), bs)
    if progress:
        it = tqdm(it)

    out = []
    for b_start in it:
        batch = slice(b_start, b_start + bs)
        tokens = torch.tensor(
            [tokenize(t, pad_to_len=pad_to_len, tokenizer=tokenizer) for t in data[batch]], dtype=torch.long
        ).to(DEVICE)
        X_tokens, y_tokens = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

        model.eval()
        logits, _ = model(X_tokens)
        log_probs = F.log_softmax(logits, dim=-1)
        y_log_probs = torch.gather(log_probs, 2, y_tokens[..., None])[..., 0]

        for i in range(y_tokens.shape[0]):
            not_pad = y_tokens[i] != tokenizer["<PAD>"]
            loss = -y_log_probs[i, not_pad].mean()
            out.append(loss.item())

    return out

def generate_text(model, tokenizer, tokenizer_inv, context="<START>", gen_tokens=10, temperature=0.6):
    context_ids = tokenize(context, tokenizer=tokenizer, include_stop=False)
    context_tensor = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        for _ in range(gen_tokens):
            logits, _ = model(context_tensor)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context_tensor = torch.cat([context_tensor, next_token], dim=1)
            if next_token.item() == tokenizer['<END>']:
                break
                
    generated = context_tensor.squeeze(0).tolist()
    return decode(generated, tokenizer , tokenizer_inv, end_at_stop=True, omit_pad=True)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.Wq(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wk(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.Wv(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V).transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        return self.Wo(output)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        logits = self.fc(x)
        return logits, None
    

def plot_losses(train_losses, val_losses, save_path='training_losses.png'):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): List of training loss values per epoch
        val_losses (list): List of validation loss values per epoch
        save_path (str): Path to save the generated plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Annotate final values
    plt.text(len(train_losses)-1, train_losses[-1], 
             f'{train_losses[-1]:.2f}', ha='left', va='center')
    plt.text(len(val_losses)-1, val_losses[-1], 
             f'{val_losses[-1]:.2f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_model(model, train_loader, val_loader, tokenizer , tokenizer_inv, epochs=10, lr=0.0001):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for X, y in tqdm(train_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            logits, _ = model(X)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits, _ = model(X)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        print(f"Sample text: {generate_text(model, tokenizer, tokenizer_inv)}")
    
    return train_losses, val_losses

def main():
    train_dataset, val_dataset, tokenizer, tokenizer_inv = load_and_preprocess_data()

    # train_dataset = Subset(train_dataset, range(50))
    # val_dataset = Subset(val_dataset, range(5))
    
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    vocab_size = len(tokenizer)
    d_model = 512
    num_heads = 8
    num_layers = 6
    ff_dim = 2048
    dropout = 0.1
    
    model = TransformerLM(vocab_size, d_model, num_heads, num_layers, ff_dim, dropout).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    train_losses, val_losses = train_model(model, train_loader, val_loader , tokenizer , tokenizer_inv)
    plot_losses(train_losses, val_losses)
    torch.save(model.state_dict(), 'transformer_lm.pth')

    # model_path = 'transformer_lm.pth'
    # test_file = '/content/shakespear_test.txt'
    # generated_texts, ppl = inference(model_path, test_file, tokenizer, tokenizer_inv)
    # print(f"\nTest perplexity: {ppl:.2f}")
    # print("Sample generated text:", generated_texts[0])

def inference(model_path, test_file, tokenizer, tokenizer_inv, gen_tokens=10, temperature=0.6):
    model = TransformerLM(len(tokenizer)).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with open(test_file, "r") as f:
        lines = f.readlines()
    
    generated_texts = []
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for line in lines:
            context = line.strip()
            generated = generate_text(model, tokenizer, tokenizer_inv, context, gen_tokens, temperature)
            generated_texts.append(generated)
            
            tokens = tokenize(context, tokenizer=tokenizer)
            tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
            X_tokens, y_tokens = tokens_tensor[:, :-1], tokens_tensor[:, 1:]
            
            logits, _ = model(X_tokens)
            log_probs = F.log_softmax(logits, dim=-1)
            y_log_probs = torch.gather(log_probs, 2, y_tokens[..., None])[..., 0]
            
            not_pad = y_tokens != tokenizer["<PAD>"]
            loss = -y_log_probs[not_pad].mean()
            total_loss += loss.item()
            count += 1
    
    perplexity = math.exp(total_loss / count)
    return generated_texts, perplexity

if __name__ == "__main__":
    main()
   
