#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
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
  for(size_t i = 0; i < length; i++) {
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

#define n_vocab 27
#define n_context 3


size_t build_data_set(char** names, size_t n_names, int **X, int **Y) {

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

/* math */

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

typedef struct matrix_t {
  int shape[2];
  float* data;
} matrix_t;

matrix_t mat_randn(int n, int m) {

  matrix_t mat = {
    .shape = {n, m},
    .data = malloc(n * m * sizeof(float))
  };

  for (int i = 0; i < n; i++) {
    for( int j = 0; j < m; j++) {
      mat.data[i * m + j] = (float)randn();
    }
  }

  return mat;
}

matrix_t mat_zeros(int n, int m) {
  return (matrix_t){
    .shape = {n, m},
    .data = calloc(n * m, sizeof(float))
  };
}

matrix_t mat_zeros_like(matrix_t t) {
 return (matrix_t){
    .shape = {t.shape[0], t.shape[1]},
    .data = calloc(t.shape[0] * t.shape[1], sizeof(float))
  };
}

matrix_t mmul(matrix_t a, matrix_t b, matrix_t out) {

  assert(a.shape[1] == b.shape[0]);
  assert(a.shape[0] == out.shape[0]);
  assert(b.shape[1] == out.shape[1]);

  for (int i = 0; i < out.shape[0]; i++) {
    for (int j = 0; j < out.shape[1]; j++) {

      float value = 0.0f;
      for (int k = 0; k < b.shape[0]; k++) {
        value += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
      }
      out.data[i * out.shape[1] + j] = value; 
    }
  }

  return out;
}

matrix_t mmuls(matrix_t a, float s) {
  for (int i = 0; i < a.shape[0]; i++) {
    for (int j = 0; j < a.shape[1]; j++) {
      a.data[i * a.shape[1] + j] *= s;
    }
  }
  return a;
}

matrix_t madd(matrix_t a, matrix_t b) {
  
  assert(a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1]);

  for (int i = 0; i < a.shape[0]; i++) {
    for (int j = 0; j < a.shape[1]; j++) {
      a.data[i * a.shape[1] + j] += b.data[i * a.shape[1] + j];
    }
  }
}

matrix_t msub(matrix_t a, matrix_t b) {
  
  assert(a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1]);

  for (int i = 0; i < a.shape[0]; i++) {
    for (int j = 0; j < a.shape[1]; j++) {
      a.data[i * a.shape[1] + j] -= b.data[i * a.shape[1] + j];
    }
  }
}


matrix_t mat_transpose(matrix_t a) {

  matrix_t trans = mat_zeros(a.shape[1], a.shape[0]);

  for (int i = 0; i < a.shape[0]; i++) {
    for (int j = 0; j < a.shape[1]; j++) {
      trans.data[j * a.shape[0] + i] = a.data[i * a.shape[1] + j];
    }
  }
  return trans;
}


void mat_free(matrix_t mat) {
  if(mat.data) free(mat.data);
}

matrix_t mat_fn(matrix_t a, float(*fn)(float)) {
   for (int i = 0; i < a.shape[0]; i++) {
    for (int j = 0; j < a.shape[1]; j++) {
      a.data[i * a.shape[1] + j] = fn(a.data[i * a.shape[1] + j]); 
    }
  }
  return a;
}

matrix_t mat_copy(matrix_t a) {
  matrix_t copy = mat_zeros_like(a);
  memcpy(copy.data, a.data, a.shape[0] * a.shape[1]);
  return copy;
}


void pmat(matrix_t a) {

  float sum = 0.0f, min = 0.0f, max = 0.0f;

  for (int i = 0; i < a.shape[0]; i++) {
    for (int j = 0; j < a.shape[1]; j++) {
      float val = a.data[i * a.shape[1] + j];
      sum += val;
      if (val < min) min = val;
      if (val > max) max = val;
    }
  }
  printf("shape(%i, %i) mean: %.2f, min: %.2f, max: %.2f\n", a.shape[0], a.shape[1], sum / (a.shape[0] * a.shape[1]), min, max);
}

#define n_embd 10
#define n_hidden 200




/* Linear layer */

typedef struct linear_t {
  matrix_t weights;
  matrix_t bias;

  matrix_t weights_grad;
  matrix_t bias_grad;

  matrix_t out;
  matrix_t in;
} linear_t;


linear_t linear_layer(int in_layer, int out_layer) {
  return (linear_t) {
    .weights = mmuls(mat_randn(in_layer, out_layer), 1 / sqrt(in_layer)),
    .bias = mmuls(mat_randn(1, out_layer), 0.01),

    .weights_grad = mat_zeros(in_layer, out_layer),
    .bias_grad = mat_zeros(1, out_layer),
    
  };
}

matrix_t linear_forward(linear_t* layer, matrix_t input) {

  if (layer->out.data == NULL) {
    layer->out = mat_zeros(input.shape[0], layer->weights.shape[1]);
  } else if (layer->out.shape[0] != input.shape[0]) {
    mat_free(layer->out);
    layer->out = mat_zeros(input.shape[0], layer->weights.shape[1]);  
  } 

  mmul(input, layer->weights, layer->out);
  for (int i = 0; i < layer->out.shape[0]; i++) {
    for (int j = 0; j < layer->out.shape[1]; j++) {
      layer->out.data[i * layer->out.shape[1] + j] += layer->bias.data[j];
    } 
  } 
  
  layer->in = input;
  return layer->out;
} 

matrix_t linear_backward(linear_t* linear, matrix_t grad) {

  matrix_t WT = mat_transpose(linear->weights);
  matrix_t input_grad = mat_zeros(grad.shape[0], WT.shape[1]); 
  mmul(grad, WT, input_grad);
  mat_free(WT);
  
  matrix_t inT = mat_transpose(linear->in);
  mmul(inT, grad, linear->weights_grad);
  mat_free(inT);

  for (int i = 0; i < grad.shape[0]; i++) {
    float sum = 0.0f;
    for (int j = 0; j < grad.shape[1]; j++) {
      sum += grad.data[i * grad.shape[1] + j];
    }
    linear->bias_grad.data[i] = sum;
  }

  return input_grad;
}


/* Tanh layer */

typedef struct tanh_t {
  matrix_t out;
} tanh_t;

matrix_t tanh_forward(tanh_t* layer, matrix_t input) {

  if (layer->out.data == NULL) {
    layer->out = mat_zeros_like(input);
  } else if (layer->out.shape[0] != input.shape[0] || layer->out.shape[1] != input.shape[1]) {
    mat_free(layer->out);
    layer->out = mat_zeros_like(input);  
  } 

  for (int i = 0; i < input.shape[0]; i++) {
    for (int j = 0; j < input.shape[1]; j++) {
      layer->out.data[i * layer->out.shape[1] + j] = tanh(input.data[i * input.shape[1] + j]);
    } 
  } 

  return layer->out;
}

matrix_t tanh_backward(tanh_t* layer, matrix_t grad) {

  matrix_t output_grad = mat_zeros_like(grad);

  for (int i = 0; i < grad.shape[0]; i++) {
    for (int j = 0; j < grad.shape[1]; j++) {
      float h = layer->out.data[i * layer->out.shape[1] + j];
      output_grad.data[i * output_grad.shape[1] + j] = (1.0 - h * h) * grad.data[i * grad.shape[1] + j];
    }
  }  

  return output_grad;
}

/* 
* Embedding layer 
*/

typedef struct emb_t {
   matrix_t embedding;
   matrix_t embedding_grad;

   matrix_t out;
   int context_n;
} emb_t;

emb_t embedding_layer(int context_length, int vocab_length, int emb_dim) {
  return (emb_t) {
    .embedding = mat_randn(vocab_length, emb_dim),
    .embedding_grad = mat_zeros(vocab_length, emb_dim),
    .context_n = context_length
  };
}

matrix_t emb_forward(emb_t* layer, int idx_n,  int* idx) {
  
  if (layer->out.data == NULL) {
    layer->out = mat_zeros(idx_n, layer->context_n * layer->embedding.shape[1]);
  } else if (layer->out.shape[0] != idx_n) {
    mat_free(layer->out);
    layer->out = mat_zeros(idx_n, layer->context_n * layer->embedding.shape[1]);
  }

  for (int i = 0; i < idx_n;    i++) {
    for (int j = 0; j < layer->context_n;  j++) {
      for (int k = 0; k < layer->embedding.shape[1];   k++) {
        layer->out.data[i * layer->context_n + j * layer->embedding.shape[1] + k] = layer->embedding.data[idx[i * layer->context_n + j] * layer->embedding.shape[0] + k];
      }
    }
  }

  return layer->out;
}

void emb_backward(emb_t* layer, matrix_t grad,  int idx_n, int* Idx) {
  memset(layer->embedding_grad.data, 0, sizeof(float) * layer->embedding_grad.shape[0] * layer->embedding_grad.shape[1]);

  matrix_t grad_view = (matrix_t) { .shape={layer->embedding_grad.shape[0], layer->embedding_grad.shape[1]}, .data=grad.data };

   for (int i = 0; i < idx_n;    i++) {
    for (int j = 0; j < layer->context_n;  j++) {
      int index = Idx[i * layer->context_n + j];
      for (int k = 0; k < layer->embedding.shape[1];   k++) {
        // layer->embedding_grad.data[i * layer->context_n + j * layer->embedding.shape[1] + k] += grad_view.data[index * grad_view.shape[0] + k];
      }
    }
  }
}

/* define model */

emb_t embedding;
linear_t hidden;
tanh_t hidden_act;
linear_t  output;


matrix_t forward(int n_samples, int* X) {
  
  matrix_t emb = emb_forward(&embedding, n_samples, X);
  // pmat(emb);
  
  // h @ W1 + b1
  matrix_t hpreact = linear_forward(&hidden, emb);
  // Tanh 
  matrix_t h = tanh_forward(&hidden_act, hpreact);
  // h @ W2 + b2 
  matrix_t out = linear_forward(&output, h);
  
  return out;
}



float cross_entropy(int n, matrix_t input, int* Y) {

  matrix_t logits = mat_copy(input);

  // normalize logits and exp
  float probssum = 0;
  for (int i = 0; i < logits.shape[0]; i++) {
    float max = 0;
    for (int j = 0; j < logits.shape[1]; j++) {
      if (logits.data[i * logits.shape[1] + j] > max) 
        max = logits.data[i * logits.shape[1] + j];

    }

    float sum = 0;
    for (int j = 0; j < logits.shape[1]; j++) {
      logits.data[i * logits.shape[1] + j] = expf(logits.data[i * logits.shape[1] + j] - max);
      sum += logits.data[i * logits.shape[1] + j];
    }

    for (int j = 0; j < logits.shape[1]; j++) {
      logits.data[i * logits.shape[1] + j] = logf(logits.data[i * logits.shape[1] + j] / sum);
    }
    probssum += logits.data[i * logits.shape[1] + Y[i]];
  }

  mat_free(logits);
  return -(probssum / n);
}

matrix_t cross_entropy_backward(int n, matrix_t logits, int* Y) {

  matrix_t dlogits = mat_copy(logits);

  // softmax
  for (int i = 0; i < dlogits.shape[0]; i++) {
    float sum = 0.0f;
    for (int j = 0; j < dlogits.shape[1]; j++) {
      sum += logits.data[i * dlogits.shape[1] + j];
    }

    for (int j = 0; j < dlogits.shape[1]; j++) {
      dlogits.data[i * dlogits.shape[1] + j] /= sum;
    }
  }

  // add loss from section occurence
  for (int i = 0; i < n; i++) {
    dlogits.data[i * dlogits.shape[1] + Y[i]] -= 1;
  }

  mmuls(dlogits, 1.0f / (float)n);

  return dlogits;
}



int main() {

  srand(42);   // Set random seed.


  // Read names in 
  int names_count = 0;
  char** names = read_names(&names_count);

  // Shuffle data set
  for (size_t i = 0; i < names_count; i++) {
    int switch_index = rand() % names_count;   
    char* tmp = names[i];
    names[i] = names[switch_index];
    names[switch_index] = tmp;
  }


  float train_split = 0.8;
  float test_split = 0.1;
  


  int* Xtr, *Ytr = NULL;
  size_t n_train = build_data_set(names, (int)(names_count * train_split), &Xtr, &Ytr); // 80%

  int* Xte, *Yte = NULL;
  size_t n_test = build_data_set(names + (int)(names_count * train_split), (int)names_count * test_split, &Xte, &Yte); // 10%

  int* Xdev, *Ydev = NULL;
  size_t n_dev = build_data_set(names + (int)(names_count * (train_split + test_split)), (int)names_count *  (1 - test_split - train_split), &Xdev, &Ydev); // 10%

  printf("== Split: train %li, test %li, dev %li\n", n_train, n_test, n_dev);

  /* model definition */

  embedding = embedding_layer(n_context, n_vocab, n_embd);

  hidden = linear_layer(n_embd * n_context, n_hidden);

  output = linear_layer(n_hidden, n_vocab);
  

  /* Training */
  
#define bs 32
#define lr 0.1
#define steps 400

  matrix_t dlogits = mat_zeros(bs, n_vocab); 


  for (int step = 0; step < steps; step++) {

    int X[bs * n_context] = { 0 };
    int Y[bs * 1        ] = { 0 };

    for (int i = 0; i < bs; i++) {
      int ix = rand() % n_train;
      for (int j = 0; j < n_context; j++) {
        X[i * n_context + j] = Xtr[ix * n_context + j];
      }
      Y[i] = Ytr[ix];
    }

    matrix_t logits = forward(bs, X);


    float loss = cross_entropy(bs, logits, Y);  

    /* backprop */

    matrix_t dlogits = cross_entropy_backward(bs, logits, Y);
    pmat(dlogits);


    matrix_t dh = linear_backward(&output, dlogits);

    matrix_t dhpreact = tanh_backward(&hidden_act, dh);

    matrix_t demb = linear_backward(&hidden, dhpreact);
    

    emb_backward(&embedding, demb, bs, X);


    madd(hidden.weights, mmuls(hidden.weights_grad, -lr));
    madd(hidden.bias, mmuls(hidden.bias_grad, -lr));
    pmat(output.weights_grad);


    
    // madd(output.weights, mmuls(output.weights_grad, -lr));
    // madd(output.bias, mmuls(output.bias_grad, -lr));

    // madd(embedding.embedding, mmuls(embedding.embedding_grad, -lr));

    mat_free(dlogits); 
    mat_free(dh); 
    mat_free(dhpreact); 
    mat_free(demb);

    

    printf("%i / %i : loss %f\n", step, steps, loss);
  
  }

  return 0;
}