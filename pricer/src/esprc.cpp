#include <iostream>
#include <set>
#include <queue>
#include <algorithm>
#include "esprc.hpp"

using std::list;
using std::queue;
using std::set;
using std::sort;
using std::vector;

using std::cout;
using std::endl;

Esprc::Esprc(int _num_customers,
             vector<vector<double>> _distances,
             vector<int> _demands,
             int _capacity)
{
    num_customers = _num_customers;
    capacity = _capacity;

    demands.assign(num_customers + 2, 0);
    for (int i = 0; i < num_customers; i++)
        demands[i + 1] = _demands[i];

    successors.assign(num_customers + 2, vector<int>());
    for (int i = 1; i <= num_customers; i++)
        successors[0].push_back(i);

    for (int i = 1; i <= num_customers; i++)
        for (int j = 1; j <= num_customers + 1; j++)
        {
            if (i == j)
                continue;
            successors[i].push_back(j);
        }

    distances.assign(num_customers + 1, vector<double>(num_customers + 2, 0));
    for (int i = 0; i < num_customers + 1; i++)
    {
        for (int j = 0; j < num_customers + 1; j++)
            distances[i][j] = _distances[i][j];
        distances[i][num_customers + 1] = distances[i][0];
    }

    init_pool_size = 1000;
    costs.assign(num_customers + 2, 0);
    labels.assign(num_customers + 2, list<Label *>());
}

Label *Esprc::get_from_pool(void)
{
    if (pool.empty())
    {
        pool.reserve(pool_size);
        for (int i = 0; i < pool_size; i++)
        {
            Label *l = new Label(num_customers + 2);
            pool.push_back(l);
        }
        pool_size *= 2;
    }

    Label *l = pool.back();
    pool.pop_back();
    return l;
}

void Esprc::return_to_pool(Label *l)
{
    pool.push_back(l);
}

Label *Esprc::extend(const Label *li, int i, int j)
{
    Label *lj = get_from_pool();
    lj->is_new = true;
    lj->cost = li->cost;
    lj->reachables = static_cast<const boost::dynamic_bitset<> &>(li->reachables);
    lj->path = li->path;
    lj->resource = li->resource;
    lj->sum_reachable = li->sum_reachable;

    double c = distances[i][j] + costs[j];
    int r = demands[j];
    lj->resource += r;
    lj->cost += c;
    lj->reachables.set(j);
    lj->path.set(j);
    lj->sum_reachable++;

    for (int k : successors[j])
    {
        int r = demands[k];
        if (lj->reachables[k])
            continue;
        if (lj->resource + r > capacity)
        {
            lj->reachables.set(k);
            lj->sum_reachable++;
        }
    }
    return lj;
}

bool Esprc::dominates(const Label &l1, const Label &l2)
{
    if (l1.cost > l2.cost)
        return false;
    if (l1.sum_reachable > l2.sum_reachable)
        return false;
    if (l1.resource > l2.resource)
        return false;
    if (!l1.reachables.is_subset_of(l2.reachables))
        return false;
    return true;
}

bool Esprc::add_label(Label *l1, int j)
{
    list<Label *> &labels = this->labels[j];
    for (auto itr = labels.begin(); itr != labels.end(); itr++)
    {
        auto l2 = *itr;
        if (dominates(*l2, *l1))
        {
            return_to_pool(l1);
            return false;
        }
        if (dominates(*l1, *l2))
        {
            return_to_pool(l2);
            itr = labels.erase(itr);
        }
    }

    labels.push_back(l1);
    return true;
}

void Esprc::solve(vector<double> duals, int k)
{
    for (int i = 0; i < num_customers; i++)
        costs[i + 1] = -duals[i];

    pool_size = init_pool_size;
    Label *l0 = get_from_pool();
    l0->is_new = true;
    l0->cost = 0;
    l0->resource = 0;
    l0->sum_reachable = 1;
    l0->reachables.set(0);
    labels[0].push_back(l0);

    set<int> S;
    queue<int> E;
    S.insert(0);
    E.push(0);

    int num_iters = 0;
    while (!E.empty())
    {
        num_iters++;
        int i = E.front();
        E.pop();
        S.erase(i);

        for (auto li : labels[i])
        {
            if (!li->is_new)
                continue;

            for (auto j : successors[i])
            {
                if (li->reachables[j])
                    continue;
                auto lj = extend(li, i, j);
                bool changed = add_label(lj, j);
                if (changed && (S.find(j) == S.end()))
                {
                    S.insert(j);
                    E.push(j);
                }
            }
            li->is_new = false;
        }
    }

    routes.clear();
    route_costs.clear();

    auto comp = [](const Label *left, const Label *right)
    { return left->cost < right->cost; };

    vector<Label *> sink_labels(labels.back().begin(), labels.back().end());
    sort(sink_labels.begin(), sink_labels.end(), comp);

    vector<Label *>::const_iterator itr = sink_labels.begin();
    for (int i = 0, s = sink_labels.size(); i < k && i < s; i++, itr++)
    {
        const Label *l = *itr;
        if (l->cost > -1e-4)
            break;

        double route_cost = l->cost;
        vector<int> route;
        for (int i = 1; i <= num_customers; i++)
            if (l->path[i])
            {
                route.push_back(i);
                route_cost -= costs[i];
            }

        routes.push_back(route);
        route_costs.push_back(route_cost);
    }

    for (auto p : pool)
        delete p;
    pool.clear();
    pool.shrink_to_fit();

    for (list<Label *> &ls : labels)
    {
        for (auto p : ls)
            delete p;
        ls.clear();
    }
}

void Esprc::get_solution(vector<vector<int>> &routes,
                         vector<double> &route_costs)
{
    routes = this->routes;
    route_costs = this->route_costs;
}