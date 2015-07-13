#include<fstream>
#include<iostream>
#include<string>
#include<algorithm>
using namespace std;

int iTrainNum = 0;
int iTestNum = 0;

int extract(ifstream* infile, ofstream *outTrain, ofstream* outTest) {
	string strLine;
	int cnt = 0;
	while(getline(*infile, strLine)) {
		if (cnt % 5 == 0) {
			*outTest << strLine << endl;
			iTestNum++;
		} else {
			*outTrain << strLine << endl;
			iTrainNum++;
		}
		cnt++;
	}
	return 0;
}

int extract(string prefix, string strFile) {
	ifstream infile(prefix + strFile);
	if (!infile) {
		cerr << "File " + prefix + strFile + " cannot be read!" << endl;
	}
	string strTrainFile = prefix + "train_" + strFile;
	string strTestFile = prefix + "test_" + strFile;
	ofstream outTrain(strTrainFile);
	ofstream outTest(strTestFile);
	extract(&infile, &outTrain, &outTest);
	infile.close();
	outTrain.close();
	outTest.close();
}

int main(int argc, char* argv[]) {
	string prefix(argv[1]);
	prefix += "/ar_";

	extract(prefix, "data.txt");

	ofstream outStat(prefix + "statistics.txt", fstream::app);
	outStat << "training examples: " << iTrainNum << endl
		<< "test examples: " << iTestNum << endl;
	outStat.close();
	
	extract(prefix, "ratings.txt");
	extract(prefix, "special.txt");
	extract(prefix, "uppercase.txt");
	extract(prefix, "help.txt");
	extract(prefix, "productId.txt");
	extract(prefix, "time.txt");
	extract(prefix, "title.txt");
	extract(prefix, "userId.txt");
	return 0;
}
