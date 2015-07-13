#include<fstream>
#include<iostream>
#include<unordered_set>
#include<unordered_map>
#include<map>
#include<algorithm>
#include<string>
#include<vector>
#include<armadillo>
#include<float.h>

#define N_FOLD 5
#define N_CLASS 5

//stemmed
#define TRAIN_ROWS 10771
#define COLS 14921 // features + 3
#define TEST_ROWS 2693
#define UPPER_DELTA 3
#define TITLE_DELTA 8

/*
//unstemmed
#define TRAIN_ROWS 3068
#define COLS 10677
#define TEST_ROWS 767
#define UPPER_DELTA 3
*/
