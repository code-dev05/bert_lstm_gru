import torch
from torch import nn
from transformers import BertModel, BertTokenizer

with open('notebooks/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

texts = text.split('\n')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BERT_LSTM_GRU(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_size=256, num_layers=2, output_size=30522, dropout=0.3):
        super(BERT_LSTM_GRU, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.embedding_dim = self.bert.config.hidden_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(self.embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        lstm_out, _ = self.lstm(bert_output.last_hidden_state)
        gru_out, _ = self.gru(lstm_out)
        
        output = self.fc(self.dropout(gru_out[:, -1, :]))
        return output

model = BERT_LSTM_GRU().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

tokenizer = model.tokenizer
tokenized_texts = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=20)
inputs = tokenized_texts["input_ids"]
labels = inputs[:, 1:].clone()
inputs = inputs[:, :-1]
labels[labels == tokenizer.pad_token_id] = -100

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')


num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(inputs.to(device), tokenized_texts['attention_mask'][:, :-1].to(device))
    
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
