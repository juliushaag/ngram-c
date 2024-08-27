
ngram: ngram.c
	gcc ngram.c -lm -o ngram

test: ngram.c
	gcc -shared -D TEST -o tests/ngram2.so -fpic ngram.c
	python tests/test_linear.py
