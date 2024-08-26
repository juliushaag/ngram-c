
test: 
	gcc -shared -D TEST -o tests/ngram2.so -fpic ngram2.c
	python tests/test_linear.py

default: ngram2

ngram2: ngram2.c
	gcc ngram2.c -lm -o ngram -g

ngram: ngram.c
	gcc ngram.c -lm -o ngram

.PHONY: test
