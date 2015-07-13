#include "common.h"

using namespace std;
using namespace arma;

class Initializer {
public:
	Initializer(Mat<int>* pTrainWordCount, vec* pTrainLabels, 
		    Mat<int>* pTestWordCount, vec* pTestLabels) :
		    _impTrainWordCount(pTrainWordCount),
		    _impTestWordCount(pTestWordCount),
		    _vpTrainLabels(pTrainLabels),
		    _vpTestLabels(pTestLabels) {};
	int readData(bool isStemmed);
	int computeMI(Mat<int>& imWordCount, vec& vLabels, unordered_map<int, double>& mapMI, int nClass);
	int selectFeaturesBySetZero(Mat<int>& imWordCount, multimap<double, int>& mapMI, int nFeatureNeeded);
	mat genTFMatrix(Mat<int>& imWordCount, vec& vLabels);

	// extract train data and Cross-validation data from input matrix
	template<class T>
	int extractCVMatrix(T& inputMatrix, T& trainMatrix, T& cvMatrix,
						int cvstart, int cvlength) {
		int iTrainRowNum = inputMatrix.n_rows - cvlength;
		int cvend = cvstart + cvlength - 1;

		trainMatrix.set_size(iTrainRowNum, inputMatrix.n_cols);
		cvMatrix.set_size(cvlength, inputMatrix.n_cols);
		cvMatrix = inputMatrix.rows(cvstart, cvend);
		if (cvstart == 0 && cvlength != inputMatrix.n_rows) {
			trainMatrix = inputMatrix.rows(cvend + 1, inputMatrix.n_rows - 1);
		} else if (cvend == inputMatrix.n_rows - 1 && cvlength != inputMatrix.n_rows) {
			trainMatrix = inputMatrix.rows(0, cvstart - 1);
		} else {
			trainMatrix.rows(0, cvstart - 1) = inputMatrix.rows(0, cvstart - 1);
			trainMatrix.rows(cvstart, iTrainRowNum - 1) = inputMatrix.rows(cvend + 1, inputMatrix.n_rows - 1);
		}
		return 0;
	}
private:
	Mat<int>* _impTrainWordCount, *_impTestWordCount;
	vec* _vpTrainLabels, *_vpTestLabels;
	int _split(string& s, string& delim, vector<string>* ret);
	int _readData(bool isStemmed, bool isTrainData);
};
