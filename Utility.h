#include "common.h"
#include<utility> 
using namespace std;

class Utility{
public:
	template<class T>
	int printMap(T map, string title = "") {
		if (title != "") {
			cout << title;
		}
		for (auto it = map.begin(); it != map.end(); it++) {
			cout << it->first << "  "<< it->second << endl;
		}
		return 0;
	}
	
	template<class K, class V>
	multimap<V, K> invertMap(map<K, V>& src) {
		multimap<V, K> dest;
		for (auto it = src.begin(); it != src.end(); it++) {
			dest.insert(pair<V, K>(it->second, it->first));
		}
		return dest;
	}

	template<class K, class V>
	multimap<V, K> invertMap(unordered_map<K, V>& src) {
		multimap<V, K> dest;
		for (auto it = src.begin(); it != src.end(); it++) {
			dest.insert(pair<V, K>(it->second, it->first));
		}
		return dest;
	}
	
	template<class K, class V>
	int printAccuracy(unordered_map<K, V> m1, unordered_map<K, V> m2, string file, string keyName, string vName1, string vName2) {
		ofstream o(file);
		o << "#" << keyName << "\t" 
			 << vName1 << "\t"
			 << vName2 << endl;
		for (auto it = m1.begin(); it != m1.end(); it++) {
			o << it->first << "\t"
			  << it->second << "\t"
			  << m2[it->first] << endl;
		}
		o.close();
		return 0;
	}
};
