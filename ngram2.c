#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

/* data */

char** read_names(int* count_out) {

  FILE* names = fopen("names.txt", "r");

  fseek(names, 0, SEEK_END);

  size_t length = ftell(names);
  
  rewind(names);

  char* names_buffer = malloc(length + 1);

  fread(names_buffer, sizeof(char), length, names);


  for(size_t i = 0; i < length; i++) {
    if (names_buffer[i] == '\n') (*count_out)++;
  }

  char** names_ptr = malloc(*count_out * sizeof(char*));

  size_t index = 1;
  for(size_t i = 0; i < length - 1; i++) {
    if (names_buffer[i] == '\n') {
      names_ptr[index++] = names_buffer + i + 1;
      names_buffer[i] = '\0';
    }
  }

  names_ptr[0] = names_buffer;

  return names_ptr;
}

int stoi(char ch) {
  if (ch == '.') return 0;
  return ch - 'a' + 1;
}

char itos(int i) {
  if (i == 0) return '.';
  return i + 'a' - 1;
}

void pbuffer(const float* buffer) {
  int n = buffer[-1];

  float sum = 0.0f;
  float min = 0.0f, max = 0.0f;

  for (int i = 0; i < n; i++) {
    float val = buffer[i];
    sum += val;
    if (val > max) max = val;
    if (val < min) min = val;
  }

  float mean = sum / (float)n;
  float std = 0.0f;
  
  for (int i = 0; i < n; i++) {
    std += (buffer[i] - mean) * (buffer[i] - mean);
  }

  std = std / (float)(n - 1);


  printf("mean: %f, std: %f, min: %f, max: %f length: %i\n", mean, sqrt(std), min, max, (int)buffer[-1]);
}


#define N_VOCAB 27


size_t build_data_set(char** names, size_t n_names, int n_context, int **X, int **Y) {

  size_t n_data_set = 0;
  for (size_t i = 0; i < n_names; i++) {
    n_data_set += strlen(names[i]) + 1;
  }

  *X = malloc(n_data_set * n_context * sizeof(int));
  *Y = malloc(n_data_set * 1         * sizeof(int));

  size_t x_index = 0;
  size_t y_index = 0;
  for (size_t i = 0; i < n_names; i++) {
    char* name = names[i];
    int name_length = strlen(name);

    assert(name_length < 128);


    int context_buffer[128] = { 0 };
    int* context = context_buffer;
    for (int j = 0; j < name_length; j++) {
      int ix = stoi(name[j]);
      for (int k = 0; k < n_context; k++) {
        (*X)[x_index++] =  context[k];
      }
      (*Y)[y_index++] = ix;

      context++;
      context[n_context - 1] = ix;
    }

    // do this additionaly for the stop token '.'
    int ix = stoi('.');
    for (int k = 0; k < n_context; k++) {
      (*X)[x_index++] =  context[k];
    }
    (*Y)[y_index++] = ix;

    context++;
    context[n_context - 1] = ix;
  }

  return n_data_set;
} 

double randn() {
  while (1) {
    double u = ((rand() / (double)RAND_MAX) * 2) - 1.0;
    double v = ((rand() / (double)RAND_MAX) * 2) - 1.0;

    double q = u*u + v*v;

    if (q == 0 || q >= 1) continue;

    double p = sqrt(-2 * log(q) / q);

    return u * p;
  }
}


void linear_forward(int B, int C, int D, 
                    const float *input, const float* weights, 
                    const float* bias, float *out) 
{
  /*
  input   (B, C)
  weights (C, D)
  bias    (1, D)
  -> out  (B, D)
  */
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < D; j++) {
      float mulacc = bias ? bias[j] : 0.0f;
      for (int k = 0; k < C; k++) {
        mulacc += input[i * C + k] * weights[k * D + j];
      }
      out[i * D + j] = mulacc;
    }   
  }
}

void linear_backward(int B, int C, int D, 
                      const float *input, const float *dout, const float *weights,
                      float *dweight, float* dbias, float* dinput)
{
  /*
  input   (B, C)
  weights (C, D)
  bias    (D)
  -> out  (B, D)
  */
  // dinput = dout @ weights^T
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < C; j++) {
      dinput[i * C + j] = 0.0f;
      for (int k = 0; k < D; k++) {
        dinput[i * C + j] += dout[i * D + k] * weights[j * D + k];
      }
    }   
  }


  /*
  input (B, C)
  dout (B, D)
  dweights (C, D)
  */
  // dweights = input^T @ dout
  for (int i = 0; i < C; i++) {
    for (int j = 0; j < D; j++) {
      dweight[i * D + j] = 0.0f; 
      for (int k = 0; k < B; k++) {
        dweight[i * D + j] += input[k * C + i] * dout[k * D + j];
      }
    }   
  }

  for (int i = 0; i < D; i++) {
    dbias[i] = 0.0;
    for (int k = 0; k < B; k++) {
      dbias[i] += dout[k * D + i];
    }
  }
}

void tanh_forward(int B, int C, const float* inp, float* out) {
  /*
  inp (B, C)
  */
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < C; j++) {
        out[i * C + j] = tanh(inp[i * C + j]);
    }
  }
} 

void tanh_backward(int B, int C, const float *doutput, const float *out, float *dinput) {
  /*
  doutput, dinput, out (B, C)
  */
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < C; j++) {
      dinput[i * C + j] = (1.0 - out[i * C + j] * out[i * C + j]) * doutput[i * C + j];
    }
  }
} 

void emb_forward(int B, int C, int I, int T, const float* emb, const int* X, float *out) {
  /*
  emb (B, C)
  X (I, T)
  out (I, C * T)
  */
  for (int i = 0; i < I; i++) {
    for (int t = 0; t < T; t++) {
      memcpy(out + i * C * T + C * t, emb + X[i * T + t] * C, C * sizeof(float));
    }
  }
}


void emb_backward(int B, int C, int I, int T, const int* X, const float* dout, float* demb){
    /*
  emb (B, C)
  X (I, T)
  dout (I, C * T)
  */ 
  memset(demb, 0, demb[-1] * sizeof(float));
  for (int i = 0; i < I; i++) {
    for (int t = 0; t < T; t++) {
      for (int k = 0; k < C; k++) {
        demb[X[i * T + t] * C + k] += dout[i * C * T + t * C + k];
      }
    }
  }
}


float cross_entropy(int B, int C, const float* input, const int* Y) {
  /*
  input (B, C)
  Y (B)
  */
  float probssum = 0.0f;

  for (int i = 0; i < B; i++) {
    float max = 0.0f;
    for (int j = 0; j < C; j++) {
      float val = input[i * C + j];
      if (val > max)
        max = val;
    }

    float sum = 0.0f; 
    for (int j = 0; j < C; j++) {
      sum += exp(input[i * C + j] - max);
    }

    probssum += log(exp(input[i * C + Y[i]] - max)  / sum);
  }
  return -(probssum / (float)B);
}

void cross_entropy_backward(int B, int C, const float* logits, const int* Y, float* dlogits) 
{
  /*
  logits (B, C)
  dlogits (B, C)
  Y (B)
  */
  for (int i = 0; i < B; i++) {
    float sum = 0.0f;
    for (int j = 0; j < C; j++) {
      sum += exp(logits[i * C + j]); 
    } 
    
    for (int j = 0; j < C; j++) {
      dlogits[i * C + j] = exp(logits[i * C + j]) / (sum);
    }

    dlogits[i * C + Y[i]] -= 1;
  }

  for (int i = 0; i < C*B; i++) {
    dlogits[i] /= (float)B;  
  }
}

void* buffers[512];
int buffer_idx = 0;

float *alloc_buffer(int n) {
  float *buffer = malloc((n + 1) * sizeof(float));
  buffers[buffer_idx++] = buffer;

  buffer[0] = n;
  buffer++;
  return buffer;
}

float* randn_buffer(int n) {
  float *buffer = alloc_buffer(n);
  for (int i = 0; i < n; i++) buffer[i] = randn();
  return buffer;
}

float* zeros_buffer(int n) {
  float *buffer = alloc_buffer(n);
  memset(buffer, 0, n * sizeof(float));
  return buffer;
}

void free_buffers() {
  for (int i = 0; i < buffer_idx; i++) {
    free(buffers[i]);
  }
}

void mul_buffer(float* buffer, float scalar) {
  int n = buffer[-1];
  for (int i = 0; i < n; i++) {
    buffer[i] *= scalar;
  }
}


void softmax(int B, int C, float* data) {
  for (int i = 0; i < B; i++) {
    float sum = 0.0f;
    for (int j = 0; j < C; j++) {
      sum += exp(data[i * C + j]); 
    } 
    
    for (int j = 0; j < C; j++) {
      data[i * C + j] = exp(data[i * C + j]) / (sum);
    }
  }
}

int multinomial_sample(const float *weights, int n) {
    float total_weight = 0.0f;
    for (int i = 0; i < n; i++) {
        total_weight += weights[i];
    }

    float random_value = ((float)rand() / RAND_MAX) * total_weight;
    
    float cumulative_weight = 0.0f;
    for (int i = 0; i < n; i++) {
        cumulative_weight += weights[i];
        if (random_value <= cumulative_weight) {
            return i;
        }
    }

    assert(0); // This should never happen
}

typedef struct ngram_model {
  
  int n_embd, n_hidden, n_context, n_batch;
  // Model parameters
  float* Emb;
  float* W1, *b1;
  float* W2, *b2;
  // Gradients
  float* dEmb;
  float* dW1, *db1;
  float* dW2, *db2;
  // Running buffers
  float *embcat;
  float *hpreact;
  float *h;
  float *logits;
  // Running grads
  float *dlogits;  
  float *dh;
  float *dhpreact;
  float *dembcat;

  void* optim_buffers;
} ngram_model;

void model_init(ngram_model* model, int n_embd, int n_hidden, int n_context) {
  
  model->Emb = randn_buffer(N_VOCAB * n_embd);

  model->W1 = randn_buffer(n_context * n_embd * n_hidden);
  mul_buffer(model->W1, (5/3) / sqrt(n_context * n_embd));

  model->b1 = randn_buffer(n_hidden);
  mul_buffer(model->b1, 0.01);

  
  model->W2 = randn_buffer(n_hidden * N_VOCAB);
  mul_buffer(model->W2, 1 / sqrt(n_hidden));

  model->b2 = randn_buffer(N_VOCAB);
  mul_buffer(model->b2, 0.01);
  
  model->dEmb = zeros_buffer(N_VOCAB * n_embd);

  model->dW1 = zeros_buffer(n_context * n_embd * n_hidden);
  model->db1 = zeros_buffer(n_hidden);
  
  model->dW2 = zeros_buffer(n_hidden * N_VOCAB);
  model->db2 = zeros_buffer(N_VOCAB);

  
  model->n_embd = n_embd;
  model->n_hidden = n_hidden;
  model->n_context = n_context;  
}

void model_prepare_run(ngram_model* model, int batch_size) {
  
  model->embcat = zeros_buffer(batch_size * model->n_context * model->n_embd);
  model->hpreact = zeros_buffer(batch_size * model->n_hidden);
  model->h = zeros_buffer(batch_size * model->n_hidden);
  model->logits = zeros_buffer(batch_size * N_VOCAB);

  
  model->dlogits = zeros_buffer(batch_size * N_VOCAB);  
  model->dh = zeros_buffer(batch_size * model->n_hidden);
  model->dhpreact = zeros_buffer(batch_size * model->n_hidden);
  model->dembcat = zeros_buffer(batch_size * model->n_context * model->n_embd);

  model->n_batch = batch_size;
}

float* forward(ngram_model* model, int* X) {

  emb_forward(N_VOCAB, model->n_embd, model->n_batch, model->n_context, model->Emb, X, model->embcat);

  linear_forward(model->n_batch, model->n_embd * model->n_context, model->n_hidden, model->embcat, model->W1, model->b1, model->hpreact);

  tanh_forward(model->n_batch, model->n_hidden, model->hpreact, model->h);

  linear_forward(model->n_batch, model->n_hidden, N_VOCAB, model->h, model->W2, model->b2, model->logits);
  
  return model->logits;
}

void backward(ngram_model* model, int* X, int* Y) {

  cross_entropy_backward(model->n_batch, N_VOCAB, model->logits, Y, model->dlogits);

  linear_backward(model->n_batch, model->n_hidden, N_VOCAB, model->h, model->dlogits, model->W2, model->dW2, model->db2, model->dh);

  tanh_backward(model->n_batch, model->n_hidden, model->dh, model->h, model->dhpreact);

  linear_backward(model->n_batch, model->n_embd * model->n_context, model->n_hidden, model->embcat, model->dhpreact, model->W1, model->dW1, model->db1, model->dembcat);

  emb_backward(N_VOCAB, model->n_embd, model->n_batch, model->n_context, X, model->dembcat, model->dEmb);

}

void optimize(ngram_model* model, float lr) {

  for(int i = 0; i < N_VOCAB * model->n_embd; i++) 
    model->Emb[i] += -lr * model->dEmb[i];

  for(int i = 0; i < model->n_context * model->n_embd * model->n_hidden; i++) 
    model->W1[i] += -lr * model->dW1[i];

  for(int i = 0; i < model->n_hidden; i++) 
    model->b1[i] += -lr * model->db1[i];

  for(int i = 0; i < model->n_hidden * N_VOCAB; i++) 
    model->W2[i] += -lr * model->dW2[i];

  for(int i = 0; i < N_VOCAB; i++) 
    model->b2[i] += -lr * model->db2[i];
}

#ifndef TEST

int main() {

  srand(42);   // Set random seed.


  // Read names in 
  int names_count = 0;
  char** names = read_names(&names_count);

  char* names_buffer = names[0];

  // Shuffle data set
  for (size_t i = 0; i < names_count; i++) {
    int switch_index = rand() % names_count;   
    char* tmp = names[i];
    names[i] = names[switch_index];
    names[switch_index] = tmp;
  }

  /* model definition */
  
  const int n_context = 5;
  const int n_embd = 16;
  const int n_hidden = 200;

  ngram_model model;
  model_init(&model, n_embd, n_hidden, n_context);

  /* Processing data */

  float train_split = 0.8;
  float test_split = 0.1;

  int* Xtr, *Ytr = NULL;
  size_t n_train = build_data_set(names, (int)(names_count * train_split), n_context, &Xtr, &Ytr); // 80%

  int* Xte, *Yte = NULL;
  size_t n_test = build_data_set(names + (int)(names_count * train_split), (int)names_count * test_split, n_context, &Xte, &Yte); // 10%

  int* Xdev, *Ydev = NULL;
  size_t n_dev = build_data_set(names + (int)(names_count * (train_split + test_split)), (int)names_count *  (1 - test_split - train_split), n_context, &Xdev, &Ydev); // 10%

  printf("== Split: train %li, test %li, dev %li\n", n_train, n_test, n_dev);

  clock_t clock_start, clock_end;
  clock_start = clock();


  /* Training */
  
  
  int bs = 32;
  float lr = 0.1;
  int steps = 10000;

  model_prepare_run(&model, bs);

  int *X = malloc(sizeof(int) * bs * n_context);
  int *Y = malloc(sizeof(int) * bs * 1        );


  float running_loss = 0.0f;

  printf("== Training for %i steps\n", steps);
  for (int step = 0; step < steps; step++) {

    for (int i = 0; i < bs; i++) {
      int ix = rand() % n_train;
      for (int j = 0; j < n_context; j++) {
        X[i * n_context + j] = Xtr[ix * n_context + j];
      }
      Y[i] = Ytr[ix];
    }
    

    float* logits = forward(&model, X);

    float loss = cross_entropy(bs, N_VOCAB, logits, Y);
    running_loss += loss;

    backward(&model, X, Y);

    optimize(&model, lr);

    if (step % 1000 == 0) {
      printf("%i / %i : loss %f\n", step, steps, running_loss / (step ? 1000 : 1));
      running_loss = 0.0f;
    }
    if (step == 10000) lr = 0.01; // lr decay
  }

  clock_end = clock();
  double cpu_time_used = ((double) (clock_end - clock_start)) / CLOCKS_PER_SEC;

  printf("== Time used %.2fs, step/s: %.2f\n", cpu_time_used, (float)steps / cpu_time_used);


  // Calculate test loss
  model_prepare_run(&model, n_test);

  float* logits = forward(&model, Xte);
  float loss = cross_entropy(n_test, N_VOCAB, logits, Yte);  

  printf("== Test loss: %f\n", loss);
  
  // Generate examples

  int n_gen = 10;
  printf("== Generating %i examples\n", n_gen);

  
  model_prepare_run(&model, 1);
  int* context = malloc(n_context * sizeof(int));

  for (int i = 0; i < n_gen; i++) {
    float index = 1.0f;
    memset(context, 0, n_context * sizeof(int));
    while (index) {

      float* logits = forward(&model, context);

      softmax(1, N_VOCAB, logits);

      index = multinomial_sample(logits, N_VOCAB);

      putchar(itos(index));
      for (int i = 0; i < n_context - 1; i++) {
        context[i] = context[i + 1];
      }
      context[n_context - 1] = index;
    }
    puts("");
  }

  free(context);

  free_buffers();
  free(X);
  free(Y);

  free(Xtr);
  free(Ytr);

  free(Xte);
  free(Yte);

  free(Xdev); 
  free(Ydev);

  free(names_buffer);
  free(names);

  return 0;
}

#endif