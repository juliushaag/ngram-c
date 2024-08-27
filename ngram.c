#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* data */

char** read_names(int* count_out) {

  FILE* names = fopen("names.txt", "r");

  if (names == NULL) return NULL;

  fseek(names, 0, SEEK_END);

  size_t length = ftell(names);
  
  rewind(names);

  char* names_buffer = malloc(length + 1);

  fread(names_buffer, sizeof(char), length, names);


  for(size_t i = 0; i < length; i++) {
    if (names_buffer[i] == '\n') (*count_out)++;
  }

  char** names_ptr = malloc(*count_out * sizeof(char*));

  size_t index = 0;
  names_ptr[index++] = names_buffer;
  for(size_t i = 0; i < length; i++) {
    if (names_buffer[i] == '\n') {
      names_ptr[index++] = names_buffer + i + 1;
      names_buffer[i] = '\0';
    }
  }

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
  #pragma omp parallel for collapse(2)
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
  #pragma omp parallel for collapse(2)
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
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < C; i++) {
    for (int j = 0; j < D; j++) {
      dweight[i * D + j] = 0.0f; 
      for (int k = 0; k < B; k++) {
        dweight[i * D + j] += input[k * C + i] * dout[k * D + j];
      }
    }   
  }

  #pragma omp parallel for
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
  #pragma omp parallel for collapse(2)
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
  #pragma omp parallel for collapse(2)
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
  #pragma omp parallel for collapse(2)
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
  memset(demb, 0, B * C * sizeof(float));
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
 
  omp_lock_t plock;
  omp_init_lock(&plock);

  float probssum = 0.0f;
  #pragma omp parallel for 
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
    
    omp_set_lock(&plock);
    probssum += log(exp(input[i * C + Y[i]] - max)  / sum);
    omp_unset_lock(&plock);
  }

  
  omp_destroy_lock(&plock);
  return -(probssum / (float)B);
}

void cross_entropy_backward(int B, int C, const float* logits, const int* Y, float* dlogits) 
{
  /*
  logits (B, C)
  dlogits (B, C)
  Y (B)
  */
  #pragma omp parallel for
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

  
  #pragma omp parallel for
  for (int i = 0; i < C*B; i++) {
    dlogits[i] /= (float)B;  
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

void model_init(ngram_model* model, int n_embd, int n_hidden, int n_context) {

  float* buffer = calloc(
    N_VOCAB * n_embd * 2 + 
    n_context * n_embd * n_hidden * 2 +
    n_hidden * 2 +
    n_hidden * N_VOCAB * 2 +
    N_VOCAB * 2,
    sizeof(float)
  );
  
  model->dEmb = buffer;
  buffer += N_VOCAB * n_embd;
  model->Emb = buffer;
  buffer += N_VOCAB * n_embd;
  for (int i = 0; i < N_VOCAB * n_embd; i++) model->Emb[i] = randn();

  model->dW1 = buffer;
  buffer += n_context * n_embd * n_hidden;
  model->W1 = buffer;
  buffer += n_context * n_embd * n_hidden;
  for (int i = 0; i < n_context * n_embd * n_hidden; i++) model->W1[i] = randn() * ((5/3) / sqrt(n_context * n_embd));

  model->db1 = buffer;
  buffer += n_hidden;
  model->b1 = buffer;
  buffer += n_hidden;
  for (int i = 0; i < n_hidden; i++) model->b1[i] = randn() * 0.01f;

  model->dW2 = buffer;
  buffer += n_hidden * N_VOCAB;
  model->W2 = buffer;
  buffer += n_hidden * N_VOCAB;
  for (int i = 0; i < n_hidden * N_VOCAB; i++) model->W2[i] = randn() / sqrt(n_hidden);

  model->db2 = buffer;
  buffer += N_VOCAB;
  model->b2 = buffer; 
  buffer += N_VOCAB;
  for (int i = 0; i < N_VOCAB; i++) model->b2[i] = randn() * 0.01;
  
  model->n_embd = n_embd;
  model->n_hidden = n_hidden;
  model->n_context = n_context;  
  model->n_batch = 0;
}

void model_prepare_run(ngram_model* model, int batch_size) {
  
  if (model->n_batch) free(model->embcat);
  

  model->n_batch = batch_size;
  if (batch_size == 0) return;

  float *buffer = calloc(
    batch_size * model->n_context * model->n_embd * 2 + 
    batch_size * model->n_hidden * 2 + 
    batch_size * model->n_hidden * 2 + 
    batch_size * N_VOCAB * 2,
    sizeof(float)
  );

  model->embcat =  buffer;
  buffer += batch_size * model->n_context * model->n_embd;
  model->dembcat = buffer;
  buffer += batch_size * model->n_context * model->n_embd;

  model->hpreact =  buffer;
  buffer += batch_size * model->n_hidden;
  model->dhpreact = buffer;
  buffer += batch_size * model->n_hidden;
  
  model->h = buffer; 
  buffer += batch_size * model->n_hidden;
  model->dh = buffer;
  buffer += batch_size * model->n_hidden;
  
  model->logits = buffer;
  buffer += batch_size * N_VOCAB;
  model->dlogits = buffer;
  buffer += batch_size * N_VOCAB;  
}

void model_free(ngram_model* model) {

  free(model->dEmb);
  model_prepare_run(model, 0); 
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
  
  #pragma omp parallel for
  for(int i = 0; i < N_VOCAB * model->n_embd; i++) 
    model->Emb[i] += -lr * model->dEmb[i];

  #pragma omp parallel for
  for(int i = 0; i < model->n_context * model->n_embd * model->n_hidden; i++) 
    model->W1[i] += -lr * model->dW1[i];

  
  #pragma omp parallel for
  for(int i = 0; i < model->n_hidden; i++) 
    model->b1[i] += -lr * model->db1[i];

  #pragma omp parallel for
  for(int i = 0; i < model->n_hidden * N_VOCAB; i++) 
    model->W2[i] += -lr * model->dW2[i];
  
  #pragma omp parallel for
  for(int i = 0; i < N_VOCAB; i++) 
    model->b2[i] += -lr * model->db2[i];
}

#ifndef TEST

int main() {

  srand(42);

  // Read names in 
  int names_count = 0;
  char** names = read_names(&names_count);

  if (names == NULL) {
    printf("Failed to read names.txt\n");
    return 0;
  }


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
  float test_split = 0.2;

  int *Xtr, *Ytr = NULL;
  size_t n_train = build_data_set(names, (int)(names_count * train_split), n_context, &Xtr, &Ytr); // 80%

  int *Xte, *Yte = NULL;
  size_t n_test = build_data_set(names + (int)(names_count * train_split), (int)names_count * test_split, n_context, &Xte, &Yte); // 10%

  printf("== Split: train %li, test %li\n", n_train, n_test);


  /* Training */
  int bs = 32;
  float lr = 0.1;
  int steps = 10000;
  printf("== Training for %i steps\n", steps);

  model_prepare_run(&model, bs);

  int *X = calloc(bs * n_context, sizeof(int));
  int *Y = calloc(bs * 1        , sizeof(int));


  float running_loss = 0.0f;
  time_t training_start;
  time(&training_start);

  for (int step = 0; step < steps; step++) {

    for (int i = 0; i < bs; i++) {
      int ix = rand() % n_train;
      memcpy(X + i * n_context, Xtr + ix * n_context, n_context * sizeof(int));
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
  
  free(X);
  free(Y);   

  free(Xtr);
  free(Ytr);

  time_t training_end;
  time(&training_end);

  double cpu_time_used = difftime(training_end, training_start);

  printf("== Time used %.2fs, steps/s: %.2f\n", cpu_time_used, (float)steps / cpu_time_used);


  // Calculate test loss
  model_prepare_run(&model, n_test);

  float* logits = forward(&model, Xte);
  float loss = cross_entropy(n_test, N_VOCAB, logits, Yte);  
  printf("== Test loss: %f\n", loss);

  free(Xte);
  free(Yte);
  
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



  model_free(&model);
  free(names_buffer);
  free(names);
  return 0;
}

#endif