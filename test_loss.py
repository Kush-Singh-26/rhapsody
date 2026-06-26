import torch
import torch.nn.functional as F
vocab_size = 49152
# 1. Uniform logits
logits = torch.zeros(1, 1024, vocab_size)
labels = torch.randint(0, vocab_size, (1, 1024))
loss1 = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
print(f"Zeros logits loss: {loss1.item()}")

# 2. Random logits
logits = torch.randn(1, 1024, vocab_size)
loss2 = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
print(f"Random logits loss: {loss2.item()}")

# 3. What gives 5.1250?
# F.cross_entropy = -x_class + log(sum(exp(x_i)))
# If loss = 5.1250, then exp(5.1250) = 168.17.
# Maybe log(vocab_size) is computed incorrectly?
