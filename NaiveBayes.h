#include "common.h"

using namespace std;
using namespace arma;

class NaiveBayes{
public:
	double train(Mat<int>& imWordCount, vec &vLabels, int nClass);
	double predict(Mat<int>& imWordCount, vec& vLabels);
private:
	//map between word frequency matrix and labels / features
	unordered_map<int, int> _mapClassToRow;
	unordered_map<int, int> _mapRowToClass;
	vec _vClassFreq;
	vec _vClassCount;
	mat _mWordFreq;
	int _reset(int nClass, int nFeatures);
};
