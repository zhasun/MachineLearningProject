#include "LogisticRegression.h"
/*
double LogisticRegression::predict(mat &mWordMatrix, Col<int>& icLabel) {
	_icPredictedLabels.set_size(icLabel.n_rows);
	for (int i = 0; i < mWordMatrix.n_rows; i++) {
		if (_sigmoid(mWordMatrix.row(i) * _vTheta) >= 0.5) {
			_icPredictedLabels(i) = 1;
		} else {
			_icPredictedLabels(i) = 0;
		}
	}
	
}
*/

double LogisticRegression::train(mat &mWordMatrix, Col<int>& icLabels) {
	_vTheta.set_size(icLabels.n_rows);
	_vTheta.fill(0);
	double dOldCost = DBL_MAX;
	double dNewCost = _cost(mWordMatrix, icLabels);
	while (abs(dOldCost - dNewCost) > _dTol) {
		dOldCost = dNewCost;
		_vTheta = _vTheta - _dStepSize * _vGrad;
		dNewCost = _cost(mWordMatrix, icLabels);
	}
	
}

// z can be a matrix
mat LogisticRegression::_sigmoid(mat& z) {
	mat mResult(z.n_rows, z.n_cols);
	for (int i = 0; i < z.n_rows; i++) {
		for (int j = 0; j < z.n_cols; j++) {
			mResult(i, j) = 1.0 / (1 + exp(-z(i,j)));
		}
	}
	return mResult;
}

double LogisticRegression::_sigmoid(double z) {
	return 1.0 / (1+exp(-z));
}
/*
double LogisticRegression::_costWithReg(mat &mWordMatrix, Col<int>& icLabels) {
	// the last row of theta corresponds to a constant feature
	int iLastVarRow = _vTheta.n_rows - 2;
	double result = _cost(mWordMatrix, icLabels) + 
			(_dLambda / (2 * mWordMatrix.n_rows)) 
			* sum(abs(_vTheta.rows(0, iLastVarRow)));
	_vGrad.rows(0, iLastVarRow) = _vGrad(0, iLastVarRow)
		+ _dLambda * _vTheta.rows(0, iLastVarRow) / mWordMtrix.n_rows;
	return result;
}
*/
//return cost
double LogisticRegression::_cost(mat &mWordMatrix, Col<int>& icLabels) {
	double result = 0;
	_vGrad.set_size(_vTheta.n_rows);

	for (int i = 0; i < mWordMatrix.n_rows; i++) {
		(mWordMatrix.row(i) * _vTheta).print();
		//mat h = _sigmoid(mWordMatrix.row(i) * _vTheta);
	//	result -= icLabels(i) * log(h);
	//	result -= (1 - icLabels(i)) * log(1 - h);
	}
	result = result / mWordMatrix.n_rows;

	for (int i = 0; i < _vTheta.n_rows; i++) {
		double dSum = 0;
		for (int j = 0; j < mWordMatrix.n_rows; j++) {
			//double h = _sigmoid(_vTheta.t() * mWordMatrix.row(j).t());
			//dSum += (h - icLabels(j)) * mWordMatrix(j,i);
		}
		_vGrad(i) = dSum / mWordMatrix.n_rows;
	}

	return result;
}
