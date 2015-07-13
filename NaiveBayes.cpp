#include "NaiveBayes.h"

// return training accuracy
double NaiveBayes::train(Mat<int>& imWordCount, vec &vLabels, int nClass) {
	_reset(nClass, imWordCount.n_cols);
	// training
	int iTopRow = 0;
	for (int iRow = 0; iRow < imWordCount.n_rows; iRow++) {
		int iLabel = vLabels(iRow);
		if (_mapClassToRow.find(iLabel) == _mapClassToRow.end()) {
			_mapClassToRow[iLabel] = iTopRow;
			iTopRow++;
		}
		for (int jCol = 0; jCol < imWordCount.n_cols; jCol++) {
			_mWordFreq(_mapClassToRow[iLabel], jCol) += imWordCount(iRow, jCol);
		}
	}

	for (auto it = _mapClassToRow.begin(); it != _mapClassToRow.end(); it++) {
		_mapRowToClass[it->second] = it->first;
	}

	_vClassCount = sum(_mWordFreq, 1);
	double dTotalClass = sum(_vClassCount);
	_vClassFreq = _vClassCount / dTotalClass;

	for (int i = 0; i < _mWordFreq.n_cols; i++) {
		_mWordFreq.col(i) = (_mWordFreq.col(i) + 1) / (_mWordFreq.n_cols + _vClassCount);
	}

	// compute training accuracy
	
	return predict(imWordCount, vLabels);	
}

// return predict accuracy
double NaiveBayes::predict(Mat<int>& imWordCount, vec& vLabels) {
	Col<int> icPredictedLabels(vLabels.n_rows);
	for (int iRow = 0; iRow < imWordCount.n_rows; iRow++) {
		vec vProbs(_mWordFreq.n_rows, fill::ones); //probability vector for classes of current example
		for (int jCol = 0; jCol < imWordCount.n_cols; jCol++) {
			// Since train set & test set are parsed together, there is
			// no unknown feature in test set.
			// Note: in feature selection phase, the weights of discarded
			// features are set to 0.
			vProbs = vProbs % pow(_mWordFreq.col(jCol), imWordCount(iRow, jCol));
		}
		vProbs = vProbs % _vClassFreq;
		double dMax = -DBL_MAX; 
		for (int i = 0; i < vProbs.n_rows; i++) {
			if (vProbs(i) > dMax) {
				dMax = vProbs(i);
				icPredictedLabels[iRow] = _mapRowToClass[i];
			}
		}
	}

	ucolvec uAccurateLabel = (icPredictedLabels == vLabels);
	double dAccurateNum = sum(uAccurateLabel);
	return dAccurateNum / vLabels.n_rows;
}

int NaiveBayes::_reset(int nClass, int nFeatures) {
	_mapClassToRow.clear();
	_mapRowToClass.clear();
	_mWordFreq.set_size(nClass, nFeatures);
	_vClassCount.clear();
	_vClassFreq.clear();
	return 0;
}
