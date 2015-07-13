#include "common.h"
#include "Initializer.h"
#include "Utility.h"
#include "LogisticRegression.h"

using namespace std;
using namespace arma;

Mat<int> imTrainWordCount(TRAIN_ROWS, COLS, fill::zeros);
Col<int> icTrainLabels(TRAIN_ROWS, fill::zeros);

Mat<int> imTestWordCount(TEST_ROWS, COLS, fill::zeros);
Col<int> icTestLabels(TEST_ROWS, fill::zeros);

/* --- for cross validation below --- */

Mat<int> imRealTrainWordCount;
Col<int> icRealTrainLabels;

Mat<int> imCVWordCount;
Col<int> icCVLabels;

// print word counts and mutual information to external file
int printFeatureInfo(multimap<double, int>& mapMI, int iIter) {
	Row<int> irWordCount(imRealTrainWordCount.n_cols);
	irWordCount = sum(imRealTrainWordCount, 0);
	string strFreqFile;
	stringstream ss;
	ss << iIter + 1;
	ss >> strFreqFile;
	strFreqFile += ".dat";
	strFreqFile = "debug/wordFreq" + strFreqFile;
	ofstream outFreq(strFreqFile);
	outFreq << "#wordID\twordCount\tMutualInfo" << endl;
	for (auto it = mapMI.begin(); it != mapMI.end(); it++) {
		outFreq << it->second << "\t\t" 
			<< irWordCount(it->second) << "\t\t" 
			<< it->first << endl;
	}
	outFreq.close();
	return 0;
}

int main(int argc, char* argv[]) {
	Utility u;
	Initializer init(&imTrainWordCount, &icTrainLabels, &imTestWordCount, &icTestLabels);
	LogisticRegression lr;
	mat d(5, 5, fill::zeros);
	//lr.train(d, imTrainWordCount);

	cout << "Reading data..." << endl;
	// set FALSE to read unstemmed data, TRUE to read stemmed data
	init.readData(false);
	int iFeatureUBound = imTrainWordCount.n_cols - 2300;

	// N_FOLD-fold cross validation
	unordered_map<int, double> mapAvgTrainAcc;
	unordered_map<int, double> mapAvgCVAcc;
	int iCVRowNum = TRAIN_ROWS / N_FOLD;

//	int iFeatureNum = iFeatureUBound;
	for (int iFeatureNum = 100; iFeatureNum < iFeatureUBound; iFeatureNum += 100) {
		double dAvgTrainAcc = 0; // average accuracy
		double dAvgCVAcc = 0;
		cout << "Running " << N_FOLD << "-fold cross validation..." 
		     << iFeatureNum << " features" << endl;
		for (int i = 0; i < N_FOLD; i++) { 
			cout << "  CV iteration " << i + 1 << endl;
			unordered_map<int, double> mapMI;
			//split train data / labels into REAL train data / labels
			//and cross validation train data / labels.
			int iStart= i * iCVRowNum;
			cout << "    Extracing data from training set..." << endl;
			init.extractCVMatrix<Mat<int>>(imTrainWordCount, 
				imRealTrainWordCount, imCVWordCount, iStart, iCVRowNum);
			init.extractCVMatrix<Col<int>>(icTrainLabels, 
				icRealTrainLabels, icCVLabels, iStart, iCVRowNum);

			// compute mutual information
			cout << "    Computing mutual information on training set..." << endl;
			init.computeMI(imRealTrainWordCount, icRealTrainLabels, mapMI, N_CLASS);
		
			// sort mutual information in ascending order
			multimap<double, int> sortedMI = u.invertMap<int, double>(mapMI);
			//feature selection with mutual information
			cout << "    Performing feature selection on training set..." << endl;
			init.selectFeaturesBySetZero(imRealTrainWordCount, sortedMI,
iFeatureNum);
//			printFeatureInfo(sortedMI, i);
			cout << "    Training classifier..." << endl;

			cout << "    Computing mutual information on CV set..." << endl;
			init.computeMI(imCVWordCount, icCVLabels, mapMI, N_CLASS);
			sortedMI = u.invertMap<int, double>(mapMI);
			cout << "    Performing feature selection on CV set..." << endl;
			init.selectFeaturesBySetZero(imCVWordCount, sortedMI, iFeatureNum);
			cout << "    Performing prediction on CV set..." << endl;
		}
		mapAvgTrainAcc[iFeatureNum] = dAvgTrainAcc / N_FOLD;
		mapAvgCVAcc[iFeatureNum] = dAvgCVAcc / N_FOLD;
	}
	u.printAccuracy<int, double>(mapAvgTrainAcc, mapAvgCVAcc, "debug/acc.dat",
			"featureNum", "trainAcc", "CVAcc");
	
	/* temp test begin */
/*	Mat<int> imTempWordCount;
	imTempWordCount << 3 << 4 << 1 << 5 << 2 << endr
			<< 1 << 2 << 0 << 1 << 1 << endr
			<< 3 << 6 << 5 << 6 << 0 << endr;
	Col<int> icTempLabels;
	icTempLabels << 1 << endr
		     << 2 << endr
		     << 0 << endr;
	unordered_map<int, double> mapMI;
	init.computeMI(imTempWordCount, icTempLabels, mapMI, 3);
	//multimap<double,int> sortedT = u.invertMap<int, double>(mapMI);
	NaiveBayes nb;
	nb.train(imTempWordCount, icTempLabels, 3);
	predict(imTempWordCount, icTempLabels);*/
	/* temp test end */
	return 0;
}
