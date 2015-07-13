#include "Initializer.h"

// split string with delims
int Initializer::_split(string& s, string& delim, vector<string>* ret) {
	size_t last = 0;
	size_t index = s.find_first_of(delim, last);
	string retElem;
	while (index != string::npos) {
		ret->push_back(s.substr(last, index - last));
		last = index + 1;
		index = s.find_first_of(delim, last);
	}

	if (index - last > 0) {
		ret->push_back(s.substr(last, index - last));
	}
	return 0;
}

// read data
int Initializer::_readData(bool isStemmed, bool isTrainData) {
	string strPrefix;
	if (isStemmed) {
		strPrefix = "stemmed/ar_";
	} else {
		strPrefix = "unstemmed/ar_";
	}

	string strTag;
	Mat<int> *impWordCount;
	vec *vpLabels;
	if (isTrainData) {
		strTag = "train";
		impWordCount = _impTrainWordCount;
		vpLabels = _vpTrainLabels;
	} else {
		strTag = "test";
		impWordCount = _impTestWordCount;
		vpLabels = _vpTestLabels;
	}

	string strSpecialFile = strPrefix + strTag + "_special.txt";
	string strStopWordsFile = strPrefix + "stat_stopwords.txt";
	string strDataFile = strPrefix + strTag + "_data.txt";
	string strLabelFile = strPrefix + strTag + "_ratings.txt";
	string strUpperCaseFile = strPrefix + strTag + "_uppercase.txt";
	string strTitleFile = strPrefix + strTag + "_title.txt";

	string strHelpFile = strPrefix + strTag + "_help.txt";
	string strProductIdFile = strPrefix + strTag + "_productId.txt";
	string strTimeFile = strPrefix + strTag + "_time.txt";
	string strUserIdFile = strPrefix + strTag + "_userId.txt";

	// read stop word list
	unordered_set<int> setStopWords;
	ifstream inStopWords(strStopWordsFile);
	if (!inStopWords) {
		cerr << strStopWordsFile + " cannot be read!" << endl;
		return -1;
	}

	int iStopNum;
	while (!inStopWords.eof()) {
		inStopWords >> iStopNum;
		setStopWords.insert(iStopNum);
	}
	inStopWords.close();

	// read train data
	ifstream inData(strDataFile);
	if (!inData) {
		cerr << strDataFile + " cannot be read!" << endl;
		return -1;
	}	

	string strLine;
	int iRow = 0;
	while(getline(inData, strLine)) {
		vector<string> vctElement;
		string delim1 = " ";
		_split(strLine, delim1, &vctElement);
		for (auto it = vctElement.begin(); it != vctElement.end(); it++) {
			vector<string> vctSingleEle;
			string delim2 = ":";
			_split(*it, delim2, &vctSingleEle);
			int f = atoi(vctSingleEle.begin()->c_str());
			int c = atoi((*(vctSingleEle.begin() + 1)).c_str());
			// remove stop words
			if (isTrainData && setStopWords.find(f) != setStopWords.end()) {
				continue;
			}
			// insert into matrix
			(*impWordCount)(iRow, f) = c;
		}
		iRow++;
	}
	inData.close();

	// read title data
	ifstream inTitle(strTitleFile);
	if (!inTitle) {
		cerr << strTitleFile + " cannot be read!" << endl;
		return -1;
	}	

	iRow = 0;
	while(getline(inTitle, strLine)) {
		vector<string> vctElement;
		string delim1 = " ";
		_split(strLine, delim1, &vctElement);
		for (auto it = vctElement.begin(); it != vctElement.end(); it++) {
			vector<string> vctSingleEle;
			string delim2 = ":";
			_split(*it, delim2, &vctSingleEle);
			int f = atoi(vctSingleEle.begin()->c_str());
			int c = atoi((*(vctSingleEle.begin() + 1)).c_str());
			// remove stop words
			if (isTrainData && setStopWords.find(f) != setStopWords.end()) {
				continue;
			}
			// insert into matrix
			(*impWordCount)(iRow, f) += c * TITLE_DELTA;
		}
		iRow++;
	}
	inTitle.close();

	// deal with uppercase words
	// add UPPER_DELTA * (Counts of uppercase) to a specified feature
	ifstream inUpperCase(strUpperCaseFile);
	if (!inUpperCase) {
		cerr << strUpperCaseFile + " cannot be read!" << endl;
		return -1;
	}
	iRow = 0;
	while(getline(inUpperCase, strLine)) {
		if (strLine.find(":") != string::npos) {
			vector<string> vctElement;
			string delim1 = " ";
			_split(strLine, delim1, &vctElement);
			for (auto it = vctElement.begin(); it != vctElement.end(); it++) {
				vector<string> vctSingleEle;
				string delim2 = ":";
				_split(*it, delim2, &vctSingleEle);
				int f = atoi(vctSingleEle.begin()->c_str());
				int c = atoi((*(vctSingleEle.begin() + 1)).c_str());
				// if it's a stop word
				if ((*impWordCount)(iRow, f) == 0) continue;
				(*impWordCount)(iRow, f) += UPPER_DELTA * c;
			}
		}
		iRow++;
	}
	inUpperCase.close();

	// read special punctuations
	ifstream inSpecial(strSpecialFile);
	if (!inSpecial) {
		cerr << strSpecialFile + " cannot be read!" << endl;
		return -1;
	}
	
	for (iRow = 0; iRow < impWordCount->n_rows; iRow++ ) {
		int s1, s2, s3;
		inSpecial >> s1 >> s2 >> s3;
		(*impWordCount)(iRow, impWordCount->n_cols - 3) = s1;
		(*impWordCount)(iRow, impWordCount->n_cols - 2) = s2;
		(*impWordCount)(iRow, impWordCount->n_cols - 1) = s3;
	};
	inSpecial.close();

	// read labels
	ifstream inLabels(strLabelFile);
	if (!inLabels) {
		cerr << strLabelFile + " cannot be read!" << endl;
		return -1;
	}	

	for (iRow = 0; iRow < impWordCount->n_rows; iRow++) {
		double d;
		inLabels >> d;
		(*vpLabels)(iRow) = d;
	}

	return 0;
}

// read data
int Initializer::readData(bool isStemmed) {
	int iReadResultTrain = 0;
	int iReadResultTest = 0;

	if (isStemmed) {
		iReadResultTrain = _readData(true, true);
		iReadResultTest = _readData(true, false);	
	} else {
		iReadResultTrain = _readData(false, true);
		iReadResultTest = _readData(false, false);
	}

	if (iReadResultTrain || iReadResultTest != 0) {
		return 1;
	}

	return 0;
}

int Initializer::computeMI(Mat<int>& imWordCount, vec& vLabels, unordered_map<int, double>& mapMI, int nClass) {
	mapMI.clear();
	unordered_map<int, int> mapTotalWordCount;
	Mat<int> imCoocurrence(nClass, imWordCount.n_cols, fill::zeros);
	unordered_map<int, int> mapClassCount;
	unordered_map<int, int> mapClassOccur;
	unordered_map<int, int> mapClassToRow;
	unordered_map<int, double> mapClassFreq;
	int iCurrentRow = 0;
	for (int i = 0; i < imWordCount.n_rows; i++) {
		int label = vLabels(i);
		for (int j = 0; j < imWordCount.n_cols; j++) {
			mapTotalWordCount[j] += imWordCount(i,j);
			if (mapClassToRow.find(label) == mapClassToRow.end()) {
				mapClassToRow[label] = iCurrentRow;
				iCurrentRow++;
			}
			imCoocurrence(mapClassToRow[label], j) += imWordCount(i,j);
			mapClassOccur[label] += imWordCount(i,j);
		}
		mapClassCount[label]++;
	}

	for (auto it = mapClassCount.begin(); it != mapClassCount.end(); it++) {
		mapClassFreq[it->first] = static_cast<double>(it->second) / imWordCount.n_rows;
	}

	for (int iTerm = 0; iTerm < imWordCount.n_cols; iTerm++) {
		mapMI[iTerm] = -DBL_MAX; 
		for (auto itLabel = mapClassToRow.begin(); itLabel != mapClassToRow.end(); itLabel++) {
			int A = imCoocurrence(itLabel->second, iTerm);
			int B = mapTotalWordCount[iTerm] - A;
			int C = mapClassOccur[itLabel->first] - A;
			int N = imWordCount.n_rows;
			double I;
			if ((A + C) * (A + B) == 0) {
				I = -DBL_MAX; 
			} else {
				I = log(static_cast<double>(A * N) / ((A + C) * (A + B)));
			}
			// max definition
			if (I > mapMI[iTerm]) {
				mapMI[iTerm] = I;
			}
		}
	}
	return 0;
}

int Initializer::selectFeaturesBySetZero(Mat<int>& imWordCount, multimap<double, int>& mapMI, int nFeatureNeeded) {
	int iFeatureUseless = imWordCount.n_cols - nFeatureNeeded;
	auto it = mapMI.rbegin();
	int iFeatureDiscarded = 0;
	while(iFeatureDiscarded < iFeatureUseless) {
		imWordCount.col(it->second).fill(0);
		iFeatureDiscarded++;
		it++;
	}
	return 0;
}

mat Initializer::genTFMatrix(Mat<int>& imWordCount, vec& vLabels) {
	Col<int> icTermCount = sum(imWordCount, 1);
	for (int i = icTermCount.n_rows - 1; i > 0; i--) {
		if (icTermCount(i) != 0) continue;
		vLabels.shed_row(i);
		imWordCount.shed_row(i);
	}
	mat mResult(imWordCount.n_rows, imWordCount.n_cols, fill::zeros);
	rowvec rvIDF(imWordCount.n_cols, fill::zeros);
	icTermCount = sum(imWordCount, 1);

	for (int i = 0; i < imWordCount.n_rows; i++) {
		for (int j = 0; j < imWordCount.n_cols; j++) {
			mResult(i,j) = static_cast<double>(imWordCount(i,j)) / icTermCount(i);
			if(imWordCount(i,j) != 0) {
				rvIDF(j)++;
			}
		}
	}

	double n = static_cast<double>(imWordCount.n_rows);
	for(int i = 0; i < rvIDF.n_cols; i++) {
		if(rvIDF(i) == 0) continue;
		rvIDF(i) = log(rvIDF(i));
	}

	for (int i = 0; i < mResult.n_rows; i++) {
		mResult.row(i) = mResult.row(i) % rvIDF;
	}

	return mResult;
}
