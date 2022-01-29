#include <vector>
#include <queue>
#include <utility>
#include <set>
#include <map>

#include "rgen.hpp"

using std::make_pair;
using std::map;
using std::pair;
using std::queue;
using std::set;
using std::vector;

vector<pair<set<int>, double> > _get_route_costs(int num_locations, int capacity, const vector<int> &demands, const vector<vector<double> > &distances)
{
    queue<set<int> > Q;

    map<pair<set<int>, int>, double> path_distances;

    for (int i = 1; i < num_locations; i++)
    {
        set<int> s;
        s.insert(i);
        Q.push(s);

        auto pair = make_pair(s, i);
        path_distances[pair] = distances[0][i];
    }

    while (!Q.empty())
    {

        set<int> path_vertices = Q.front();
        Q.pop();

        int consumed_capacity = 0;
        for (auto v : path_vertices)
            consumed_capacity += demands[v];

        for (int i = *(path_vertices.rbegin()) + 1; i < num_locations; i++)
            if (demands[i] <= capacity - consumed_capacity)
            {
                set<int> next_path = path_vertices;
                next_path.insert(i);
                Q.push(next_path);

                for (auto last_vertex : next_path)
                {
                    set<int> prefix(next_path);
                    prefix.erase(last_vertex);

                    double min_dist = -1;
                    for (auto v : prefix)
                    {
                        double prefix_lenght = path_distances[make_pair(prefix, v)];
                        double path_length = prefix_lenght + distances[v][last_vertex];
                        if (min_dist < 0 || path_length < min_dist)
                            min_dist = path_length;
                    }

                    auto pair = make_pair(next_path, last_vertex);
                    path_distances[pair] = min_dist;
                }
            }
    }

    vector<pair<set<int>, double> > cycle_lengths;
    set<set<int> > calculated_cycles;
    for (auto p : path_distances)
    {
        set<int> path = p.first.first;
        if (calculated_cycles.find(path) == calculated_cycles.end())
        {
            double min_dist = -1;
            for (auto v : path)
            {
                double path_dist = path_distances[make_pair(path, v)];
                double cycle_dist = path_dist + distances[v][0];
                if (min_dist < 0 || cycle_dist < min_dist)
                    min_dist = cycle_dist;
            }
            calculated_cycles.insert(path);
            cycle_lengths.push_back(make_pair(path, min_dist));
        }
    }

    return cycle_lengths;
}
