#include <vector>
#include <list>
#include <boost/dynamic_bitset.hpp>

using std::list;
using std::vector;

class Label
{
public:
    Label(int num_bits)
    {
        is_new = false;
        resource = 0;
        cost = 0;
        sum_reachable = 0;
        reachables = boost::dynamic_bitset<>(num_bits);
        path = boost::dynamic_bitset<>(num_bits);
    }
    Label(bool n, int r, double c, int s,
          const boost::dynamic_bitset<> &rs,
          const boost::dynamic_bitset<> &p) : is_new(n), resource(r), cost(c), sum_reachable(s), reachables(rs), path(p) {}
    bool is_new;
    int resource;
    double cost;
    int sum_reachable;
    boost::dynamic_bitset<> reachables;
    boost::dynamic_bitset<> path;
};

class Esprc
{

public:
    Esprc(){}
    Esprc(int _num_customers,
          vector<vector<double>> _distances,
          vector<int> _demands,
          int _capacity);
    void solve(vector<double> duals, int k);
    void get_solution(vector<vector<int>>&, vector<double>&);

private:
    Label *get_from_pool(void);
    void return_to_pool(Label *l);
    Label *extend(const Label *li, int i, int j);
    bool dominates(const Label &l1, const Label &l2);
    bool add_label(Label *l1, int j);

    vector<vector<double>> distances;
    int num_customers;
    vector<int> demands;
    int capacity;
    vector<vector<int>> successors;
    vector<Label *> pool;
    int pool_size;
    int init_pool_size;
    vector<double> costs;
    vector<list<Label *>> labels;
    vector<double> route_costs;
    vector<vector<int>> routes;
};