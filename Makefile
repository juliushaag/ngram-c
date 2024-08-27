
ngram: ngram.c
	gcc ngram.c  -lgomp -lm -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes -fopenmp -o ngram

test: ngram.c
	gcc -shared -D TEST -o tests/ngram2.so -fpic ngram.c
	python tests/test_linear.py
