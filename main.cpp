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
#include <map>
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
    long id, parent;
    bool hasData;

    inline void _insert(const Point &data_in){ dataArr.push_back(data_in); }

    friend class Knn;

    public:
    Node(long id_in):id(id_in), hasData(false){ }

    Node(const Point &data_in, int id_in):id(id_in),parent(-1), hasData(true){ dataArr.push_back(data_in); }


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
    long _computeKey(unsigned int , unsigned int , unsigned int );
    long _getCellId(const Point &);
    void _flush_buffer();
    void _assemble();
    void _post_order_walk();
    string getLinearTree() const;
    string getLocalTree() const;
    public:
    Knn(){};
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
            localArr.push_back(Node(cellId)); 
            nodeTable[cellId] = localArr.size() - 1;
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
        result += "];";
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
                    localArr.back().childset.insert(make_pair(localArr[i].id, i));
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
    return 0;
}
