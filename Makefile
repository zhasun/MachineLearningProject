CC = g++-4.8
FLAG = -O2 -larmadillo
DEBUG = -g -O2 larmadillo

debug: Initializer.cpp Initializer.h main_svm.cpp common.h Utility.h
	$(CC) -std=c++11 -g -o svm $(FLAG) common.h Utility.h svm.h svm.cpp Initializer.h Initializer.cpp main_svm.cpp
lr: LogisticRegression.o Initializer.o main_lr.cpp common.h Utility.h
	$(CC) -std=c++11 -o lr $(FLAG) main_lr.cpp common.h Utility.h LogisticRegression.o Initializer.o
svm: svm.o Initializer.o main_svm.cpp common.h Utility.h
	$(CC) -std=c++11 -o svm $(FLAG) main_svm.cpp common.h Utility.h svm.o Initializer.o
nb: NaiveBayes.o Initializer.o main_nb.cpp common.h Utility.h
	$(CC) -std=c++11 -g -o nb $(FLAG) main_nb.cpp common.h Utility.h NaiveBayes.o Initializer.o
NaiveBayes.o : NaiveBayes.h NaiveBayes.cpp
	$(CC) -std=c++11 -c NaiveBayes.cpp $(FLAG)
svm.o: svm.h svm.cpp
	$(CC) -std=c++11 -c svm.cpp $(FLAG)
LogisticRegression.o: LogisticRegression.h LogisticRegression.cpp
	$(CC) -std=c++11 -c LogisticRegression.cpp $(FLAG)
Initializer.o: Initializer.h Initializer.cpp
	$(CC) -std=c++11 -c Initializer.cpp $(FLAG)
clean:
	rm *.o
