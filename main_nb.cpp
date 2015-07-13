#include "common.h"
#include "Initializer.h"
#include "Utility.h"
#include "NaiveBayes.h"

using namespace std;
using namespace arma;

Mat<int> imTrainWordCount(TRAIN_ROWS, COLS, fill::zeros);
vec vTrainLabels(TRAIN_ROWS, fill::zeros);

Mat<int> imTestWordCount(TEST_ROWS, COLS, fill::zeros);
vec vTestLabels(TEST_ROWS, fill::zeros);

/* --- for cross validation below --- */

Mat<int> imRealTrainWordCount;
vec vRealTrainLabels;

Mat<int> imCVWordCount;
vec vCVLabels;

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
	Initializer init(&imTrainWordCount, &vTrainLabels, &imTestWordCount, &vTestLabels);
	NaiveBayes nb;

	cout << "Reading data..." << endl;
	// set FALSE to read unstemmed data, TRUE to read stemmed data
	init.readData(true);

	int iFeatureUBound = imTrainWordCount.n_cols - 3000;

	// N_FOLD-fold cross validation
	unordered_map<int, double> mapAvgTrainAcc;
	unordered_map<int, double> mapAvgCVAcc;
	int iCVRowNum = TRAIN_ROWS / N_FOLD;

	int iFeatureNum = 4900;
	for (int iFeatureNum = 4000; iFeatureNum < 8001/*iFeatureUBound*/; iFeatureNum += 500) {
		double dAvgTrainAcc = 0; // average accuracy
		double dAvgCVAcc = 0;
		cout << "Running " << N_FOLD << "-fold cross validation..." 
		     << iFeatureNum << " features" << endl;
		for (int i = 0; i < 1; i++) { 
			cout << "  CV iteration " << i + 1 << endl;
			unordered_map<int, double> mapMI;
			//split train data / labels into REAL train data / labels
			//and cross validation train data / labels.
			int iStart= i * iCVRowNum;
			cout << "    Extracing data from training set..." << endl;
			init.extractCVMatrix<Mat<int>>(imTrainWordCount, 
				imRealTrainWordCount, imCVWordCount, iStart, iCVRowNum);
			init.extractCVMatrix<vec>(vTrainLabels, 
				vRealTrainLabels, vCVLabels, iStart, iCVRowNum);

			// compute mutual information
			cout << "    Computing mutual information on training set..." << endl;
			init.computeMI(imTrainWordCount, vTrainLabels, mapMI, N_CLASS);
		
			// sort mutual information in ascending order
			multimap<double, int> sortedMI = u.invertMap<int, double>(mapMI);
			//feature selection with mutual information
			cout << "    Performing feature selection on training set..." << endl;
			init.selectFeaturesBySetZero(imTrainWordCount, sortedMI,
iFeatureNum);
			//printFeatureInfo(sortedMI, i);
			cout << "    Training classifier..." << endl;
			dAvgTrainAcc += nb.train(imTrainWordCount, vTrainLabels, N_CLASS);

			cout << "    Computing mutual information on CV set..." << endl;
			init.computeMI(imTestWordCount, vTestLabels, mapMI, N_CLASS);
			sortedMI = u.invertMap<int, double>(mapMI);
			cout << "    Performing feature selection on CV set..." << endl;
			init.selectFeaturesBySetZero(imTestWordCount, sortedMI, iFeatureNum);
			cout << "    Performing prediction on CV set..." << endl;
			dAvgCVAcc += nb.predict(imTestWordCount, vTestLabels);
		}
		mapAvgTrainAcc[iFeatureNum] = dAvgTrainAcc / N_FOLD;
		mapAvgCVAcc[iFeatureNum] = dAvgCVAcc / N_FOLD;
	}
	u.printAccuracy<int, double>(mapAvgTrainAcc, mapAvgCVAcc, "debug/acc.dat",
			"featureNum", "trainAcc", "TestAcc");
	
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
