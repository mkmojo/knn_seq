#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <bitset>
#include <set>
#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <map>
#include <cmath>
using namespace std;

typedef unordered_set<int> NodeSet;
typedef map<long, int> NodeOrderSet;

class Knn;

class Point
{
    double mass;
    public:
    double x, y, z;
    long id;
    Point(istringstream &iss):id(0){
        iss >> x >> y >> z >> mass;
    }
};

class Node
{
    std::vector<Point> dataArr;
    NodeOrderSet childset;
    NodeSet childindexset;
    long id;
	int parent;
    bool hasData;
	int pointNum;
    bool isleaf;
	unsigned int level;
	double x1, x2, y1, y2, z1, z2;

    vector<vector<pair<double, long> > > neighbour;

    inline void _insert(const Point &data_in){ dataArr.push_back(data_in); }

    friend class Knn;

    public:
    Node(long id_in):id(id_in), hasData(false), isleaf(false), pointNum(0){ }

    Node(const Point &data_in, int id_in):id(id_in),parent(-1), hasData(true), isleaf(false), pointNum(0){ dataArr.push_back(data_in); }


    ~Node(){ }

    inline int getCount() const { return dataArr.size(); }

    long getId() const{
        return id;
    }

    void setId(long id_in){
        id=id_in;
    }

    int getParent() const{
        return parent;
    }

    void setParent(int parent_in){
        parent=parent_in;
    }

};

class Knn {
    vector<Point> localBuffer;
    vector<Node> localArr;
    vector<Node> localStruct;
    unordered_map<long, int> nodeTable;
    unsigned int maxlevel;
    unsigned int ndim;
    int K;
    NodeSet MyGenerate(const Node &);
    void MyCombine(Node &, const Node &);
    double large_dist_compute(const Node&, const Node&);
    double small_dist_compute(const Node&, const Node&);
    double dist_compute(double, double, double, double, double, double);
	void coordinate(Node&);
    long _computeKey(unsigned int , unsigned int , unsigned int );
    long _getCellId(const Point &);
    void _flush_buffer();
    void _assemble();
    void _post_order_walk();
    string getLinearTree() const;
    string getLocalTree() const;
    public:
    Knn(): K(3){};
	void Compute();
    void load(string);
    void build(){
        _flush_buffer();
        _assemble();
    }
    void print(){
        cout << getLinearTree() <<endl;
        cout << getLocalTree() <<endl;
    };


};



void Knn::Compute(){
    for(int i = 0; i < localStruct.size(); i++){
		//cout<<"Compute "<<i<<" "<<localStruct[i].dataArr.size()<<endl;
        if(localStruct[i].isleaf){
			auto mynode = localStruct[i];
            auto set = MyGenerate(mynode);
			
			//cout<<"leaf "<<endl;            
			//cout<<"generate end"<<endl;
			//cout<<"nn size "<<mynode.neighbour.size()<<endl;

			mynode.neighbour.resize(mynode.dataArr.size());
            for(auto node_id : set){
				if(node_id != mynode.getId()){
					auto node = localStruct[nodeTable[node_id]];
                	MyCombine(mynode, node);
				}
            }
			//cout<<"nn size 2 "<<mynode.neighbour.size()<<endl;
            for(int j = 0; j < mynode.neighbour.size(); j++){
                auto &nn = mynode.neighbour[j];
   			    sort(nn.begin(), nn.end());
                nn.resize(K);

				cout<< "*********"<<endl;
				cout<<"node id "<<mynode.getId()<<endl;
				cout<<nn[0].first<<" "<<nn[0].second<<" "<<nn[1].first<<" "<<nn[1].second<<" "<<nn[2].first<<" "<<nn[2].second<<endl;
				cout<<endl;
			}
        }
    }
}


NodeSet Knn::MyGenerate(const Node& mynode){
    NodeSet myset;
    
    Node cur = mynode;
    long id = cur.getId();
    int parent;
    multiset<double> large_dist_set;
    double large_dist;
    double small_dist;

	//cout<<"in generate"<<endl;

	stack<Node> st;

	//cout<<"======"<<endl;
	//cout<<id<<endl;
	//cout<<endl;

    while(id != 1){
		st.push(cur);
        parent = cur.getParent();
        cur = localStruct[parent];
        id = cur.getId();
		//cout<<id<<endl;
    }

    deque<Node> Q;
    deque<Node> New; 
    Q.push_back(cur);

	//cout<<"looking for knn"<<endl;

    do{
		//cout<<"do while"<<Q.size()<<" cur id "<<cur.getId()<<endl;
        //Find the largest dist
        for(int i = 0; i < Q.size(); i++){
            large_dist_set.insert(large_dist_compute(cur, Q[i]));
        }
        if(Q.size() < K+1){
            for(auto &&node : Q){
				if(node.isleaf){
					New.push_back(node);
				}
				else{
                	for(auto &&ch : node.childset){
                    	New.push_back(localStruct[ch.second]);
                	}
				}
            }
        }
		else{
        	multiset<double>::iterator i = large_dist_set.begin();
        	for(int j = 0; j < K+1; j++) i++;
        	large_dist = *i;

        	//Find the cube to be tested in next level
        	for(auto &node : Q){
            	if(small_dist_compute(cur, node) < large_dist){
					if(node.isleaf){
						New.push_back(node);
					}
					else{
                		for(auto &&ch_index : node.childset){
                    		auto child = localStruct[ch_index.second];
                    		//if(small_dist_compute(cur, child) < large_dist){
                        		New.push_back(child);
                    		//}
                		}
					}
            	}
        	}
		}
        Q.clear();
        Q = New;
        New.clear();

		cur = st.top();
		st.pop();		

    }while(!cur.isleaf);

	//cout <<"find knn"<<endl;

    for(auto&& node : Q){
        myset.insert(node.getId());
    }
    return myset;
}


double Knn::large_dist_compute(const Node& a, const Node& b){
	double ax = 0.0, ay = 0.0, az = 0.0, bx = 0.0, by = 0.0, bz = 0.0;
	if(fabs(a.x1 - b.x2) > fabs(a.x2 - b.x1)){
		ax = a.x1;
		bx = b.x2;
	}
	else{
		ax = a.x2;
		bx = b.x1;
	}
	if(fabs(a.y1 - b.y2) > fabs(a.y2 - b.y1)){
		ay = a.y1;
		by = b.y2;
	}
	else{
		ay = a.y2;
		by = b.y1;
	}
	if(fabs(a.z1 - b.z2) > fabs(a.z2 - b.z1)){
		az = a.z1;
		bz = b.z2;
	}
	else{
		az = a.z2;
		bz = b.z1;
	}

    return dist_compute(ax, ay, az, bx, by, bz);
}

double Knn::small_dist_compute(const Node& a, const Node& b){
	double ax = 0.0, ay = 0.0, az = 0.0, bx = 0.0, by = 0.0, bz = 0.0;
	if(fabs(a.x1 - b.x2) < fabs(a.x2 - b.x1)){
		ax = a.x1;
		bx = b.x2;
	}
	else{
		ax = a.x2;
		bx = b.x1;
	}
	if(fabs(a.y1 - b.y2) < fabs(a.y2 - b.y1)){
		ay = a.y1;
		by = b.y2;
	}
	else{
		ay = a.y2;
		by = b.y1;
	}
	if(fabs(a.z1 - b.z2) < fabs(a.z2 - b.z1)){
		az = a.z1;
		bz = b.z2;
	}
	else{
		az = a.z2;
		bz = b.z1;
	}


    return dist_compute(ax, ay, az, bx, by, bz);
}

void Knn::MyCombine(Node& a,  const Node& b){
	//cout<<"begin combine "<<a.dataArr.size() << " "<<b.dataArr.size()<<endl;
    for(int i = 0; i < a.dataArr.size(); i++){
        auto p = a.dataArr[i];
		auto &tmp = a.neighbour[i];
        for(int j = 0; j < b.dataArr.size(); j++){
			auto p2 = b.dataArr[j];
            tmp.push_back(make_pair(dist_compute(p.x, p.y, p.z, p2.x, p2.y, p2.z), p2.id));
        }
    }
	//cout<<"combine end "<<a.neighbour.size()<<endl;
}

void Knn::coordinate(Node& a){
	auto id = a.getId();
	int mask = 1;
	int x = 0;
	int y = 0;
	int z = 0;

	for(unsigned int i = 0; i < a.level; i++){
		x << 1;
		y << 1;
		z << 1;
		if(id & mask) z += 1;
		id >> 1;
		if(id & mask) y += 1;
		id >> 1;
		if(id & mask) x += 1;
	}

	double len = 1.0 / (1 << (a.level));
	a.x1 = x * len;
	a.y1 = y * len;
	a.z1 = z * len;
	a.x2 = a.x1 + len;
	a.y2 = a.y1 + len;
	a.z2 = a.z1 + len;

	
}

double Knn::dist_compute(double x1, double y1, double z1, double x2, double y2, double z2){
	double dist = sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2) + pow((z1 - z2), 2));
	return dist;
}

void Knn::load(string filename){
    std::ifstream fs(filename.c_str());
    std::string line;
    bool datastart = false;

    while (std::getline(fs, line)) {
        std::istringstream iss(line);
        std::string params;
        if (!datastart) {
            iss >> params;
            if (params == "NDIM")
                iss >> ndim;
            else if (params == "MAX_LEVEL")
                iss >> maxlevel;
            else if (params == "DATA_TABLE") {
                datastart = true;
                continue;
            }else if (params.empty())
                continue;
            else{
                cout << "Parse input error" <<endl;
                exit(1);
            }
        } else localBuffer.push_back(Point(iss));
    }

    //set up the id inside vec points
    for(auto &it : localBuffer){ 
        it.id = _getCellId(it);
        cout << it.id << endl;
    }
}

long Knn::_computeKey(unsigned int x, unsigned int y, unsigned z) {
    // number of bits required for each x, y and z = height of the
    // tree = level_ - 1
    // therefore, height of more than 21 is not supported
    unsigned int height = maxlevel - 1;

    long key = 1; //initialized with the leading 1
    long x_64 = (long) x << (64 - height);
    long y_64 = (long) y << (64 - height);
    long z_64 = (long) z << (64 - height);
    long mask = 1L << 63; // leftmost bit is 1, rest 0

    for (unsigned int i = 0; i < height; ++i) {
        key = key << 1;
        if (x_64 & mask) key = key | 1;
        x_64 = x_64 << 1;

        key = key << 1;
        if (y_64 & mask) key = key | 1;
        y_64 = y_64 << 1;

        key = key << 1;
        if (z_64 & mask) key = key | 1;
        z_64 = z_64 << 1;
    }
    return key;
}

long Knn::_getCellId(const Point &p) {
    //return the cell id for a point;
    //r stands for range
    double x = p.x;
    double y = p.y;
    double z = p.z;

    unsigned int numCells = 1 << (maxlevel - 1);
    double cellSize = 1.0 / numCells;

    int xKey = (x / cellSize);
    int yKey = (y / cellSize);
    int zKey = (z / cellSize);

    return _computeKey(xKey, yKey, zKey);
}

void Knn::_flush_buffer() {
    //insert points to LocalArr from lcoalBuffer
    //use nodeTable to do book keeping
    for (auto it : localBuffer) {
        long cellId = it.id;
        auto itt = nodeTable.find(cellId);
        if (itt == nodeTable.end()){
            Node new_node(cellId);
            new_node.isleaf = true;
			new_node.dataArr.push_back(it);
			new_node.hasData = true;
			new_node.pointNum = 1;
			new_node.level = maxlevel - 1;
			coordinate(new_node);
            localArr.push_back(new_node); 
            nodeTable[cellId] = localArr.size() - 1;
        }
		else{
			auto &this_node = localArr[itt->second];
			this_node.dataArr.push_back(it);
			this_node.hasData = true;
			this_node.pointNum++;
		}
    }
    cout << "DEBUG " << ":  " << "localArr size: " << localArr.size() <<endl;
    cout << "DEBUG " << ":  " << "nodeTable size: " << nodeTable.size() <<endl;
}

void Knn::_post_order_walk() {
    stack<int> aux;
    auto track = new vector<NodeOrderSet::iterator>;

    //need to repopulate the node table
    nodeTable.clear();

    for (auto&& it : localArr)
        track->push_back(it.childset.begin());

    int n = localArr.size();
    int last_visited = -1, current = n-1, peek = 0, count = 0;

    localStruct.reserve(n);
    vector<int> tab(n, 0);
    while (!aux.empty() || current != -1) {
        if (current != -1) {
            aux.push(current);
            current = (*track)[current]->second;
        } else {
            peek = aux.top();
            ++(*track)[peek];
            if ((*track)[peek] != localArr[peek].childset.end()
                    && last_visited != ((*track)[peek])->second) {
                current = (*track)[peek]->second;
            } else {
                localStruct.push_back(localArr[peek]);
                nodeTable[localArr[peek].id] = localStruct.size()-1;
                tab[peek] = count++;
                last_visited = aux.top();
                aux.pop();
            }
        }
    }
    delete track;

    //clear localArr. localStruct will be used from now on
    localArr.clear();

    //fix the parent and children links
    for(auto&& it : localStruct){
        it.parent = tab[it.parent];

        for(auto&& itt : it.childset){
            itt.second = tab[itt.second];
            it.childindexset.insert(itt.second);
        }
    }
}

string Knn::getLocalTree() const {
    std::string result("");
    for(auto&& it : this->localStruct){
        auto bset = std::bitset<7>(it.getId());
        result = result + bset.to_string() + "*";
    }
    if(!result.empty()) result.pop_back();
    return result;
}

string Knn::getLinearTree() const {
    std::string result("");
    for(auto it = this->localStruct.rbegin(); it != this->localStruct.rend(); it++){
        auto curr_bset = std::bitset<7>((*it).id);
        result = result + curr_bset.to_string() + "(" + std::to_string((*it).getCount()) + ")" + "[";
        for(auto itt = (*it).childset.begin(); itt != (*it).childset.end(); itt++){
            if((*itt).second != -1){
                auto child_bset = std::bitset<7>(this->localStruct[(*itt).second].id);
                result = result + child_bset.to_string() + ",";
            }
        }
        if(result.back() == ',') result.pop_back();
        result += "];\n";
    }
    if(!result.empty()) result.pop_back();
    return result;
}

void Knn::_assemble() {
    set<int> aux;
    int old_last = -1;
    int new_last = localArr.size()-1;

    while(new_last - old_last > 1){ 
        for(int i = old_last+1; i <= new_last; i++)
            aux.insert(localArr[i].id >> 3);

        old_last = localArr.size()-1;

        for(auto it : aux){
            localArr.push_back(Node(it));
            for(int i = 0; i < localArr.size()-1; i++){
                if((localArr[i].id) >> 3 == it){
					auto &this_node = localArr.back();
                    this_node.childset.insert(make_pair(localArr[i].id, i));
					this_node.pointNum += localArr[i].pointNum;
					this_node.level = localArr[i].level - 1;
					coordinate(this_node);
                    localArr[i].parent = localArr.size()-1;
                }
            }
        }
        aux.clear();
        new_last = localArr.size()-1;
    }

    localArr.back().parent = -1;
    for(auto&& it : localArr){
        if(it.childset.empty())
            it.childset.insert(std::make_pair(-1,-1));
    }

    cout << "DEBUG after assemble " << ":  " << "localArr size: " << localArr.size() <<endl;
    cout << "DEBUG after assemble " << ":  " << "nodeTable size: " << nodeTable.size() <<endl;

    _post_order_walk();
}





int main(int argc, char** argv)
{
    Knn knn;
    knn.load(string(argv[1]));
    knn.build();
    knn.print();
	cout<<"compute begin"<<endl;
	knn.Compute();
	cout<<"compute end"<<endl;
    return 0;
}
