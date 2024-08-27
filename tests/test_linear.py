import ctypes
import torch
import torch.nn.functional as F 
import numpy as np
torch.manual_seed(42)

def tc(tensor : torch.Tensor):
  return tensor.detach().numpy().ctypes

ngram = ctypes.CDLL("tests/ngram2.so")

"""
TEST FORWARD PASS
"""

# Test cross entropy
n_emb = 5
n_vocab = 27
EMB = torch.randn((n_vocab, n_emb), requires_grad=True)

n_batch = 32
n_context = 20
X = torch.randint(0, n_vocab, (n_batch, n_context), dtype=torch.int32)
Y = torch.randint(0, n_vocab, (n_batch,), dtype=torch.int32)

out = EMB[X]
out = out.view(out.shape[0], -1) 

out_buffer = torch.zeros_like(out)

ngram.emb_forward(n_vocab, n_emb, n_batch, n_context, tc(EMB), tc(X), tc(out_buffer))

assert torch.allclose(out, out_buffer), out_buffer.mean()


n_hidden = 200
W1 = torch.randn((n_context * n_emb, n_hidden), requires_grad=True)
b1 = torch.randn((n_hidden), requires_grad=True)

hpreact = out @ W1 + b1

hpreact_buffer = torch.zeros_like(hpreact)
ngram.linear_forward(n_batch, n_context * n_emb, n_hidden, tc(out), tc(W1), tc(b1), tc(hpreact_buffer))

assert torch.allclose(hpreact, hpreact_buffer, atol=1e-5)

h = torch.tanh(hpreact)
hout = torch.zeros_like(h)

ngram.tanh_forward(h.shape[0], h.shape[1], tc(hpreact), tc(hout))

assert torch.allclose(h, hout)

W2 = torch.randn((n_hidden, n_vocab), requires_grad=True)
b2 = torch.randn((n_vocab), requires_grad=True)

logits = h @ W2 + b2
logits_buffer = torch.zeros_like(logits)

ngram.linear_forward(n_batch, n_hidden, n_vocab, tc(h), tc(W2), tc(b2), tc(logits_buffer))
assert torch.allclose(logits, logits_buffer, atol=1e-5)


loss = F.cross_entropy(logits, Y.to(dtype=torch.long))

ngram.cross_entropy.restype = ctypes.c_float

assert torch.allclose(torch.tensor(ngram.cross_entropy(n_batch, n_vocab, tc(logits), tc(Y))),  loss), "cross_entriopy forward pass failed"

loss.backward()


dlogits = torch.zeros_like(logits)

logit_grad = F.softmax(logits, 1)
logit_grad[range(n_batch), Y] -= 1
logit_grad /= n_batch

ngram.cross_entropy_backward(n_batch, n_vocab, tc(logits), tc(Y), tc(dlogits))

assert torch.allclose(dlogits, logit_grad)


dW2 = torch.zeros_like(W2)
db2 = torch.zeros_like(b2)
dh = torch.zeros_like(h)

ngram.linear_backward(n_batch, n_hidden, n_vocab, tc(h), tc(dlogits), tc(W2), tc(dW2), tc(db2), tc(dh))

assert torch.allclose(db2, b2.grad)
assert torch.allclose(dW2, W2.grad)
