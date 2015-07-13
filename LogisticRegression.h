#include "common.h"

using namespace std;
using namespace arma;

class LogisticRegression {
public:
	double train(mat &mWordMatrix, Col<int>& icLabels);
	double predict(mat &mWordMatrix, Col<int>& icLabels);
private:
	double _cost(mat &mWordMatrix, Col<int>& icLabels);
	mat _sigmoid(mat &z);
	double _sigmoid(double z);

	vec _vGrad;
	vec _vTheta;
	Col<int> _icPredictedLabels;
	double _dStepSize = 0.001;
	double _dTol = 0.1;
	double _dLambda =1111; 
};
