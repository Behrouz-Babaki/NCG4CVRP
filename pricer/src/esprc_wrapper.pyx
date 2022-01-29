from libcpp.vector cimport vector

cdef extern from "esprc.hpp":
    cdef cppclass Esprc:
        Esprc()
        Esprc(int, vector[vector[double]], 
              vector[int], int) except +
        void solve(vector[double], int)
        void get_solution(vector[vector[int]]&, vector[double]&)


cdef class ESPRC:
    cdef Esprc solver

    def __init__(self, int num_customers,
                 vector[vector[double]] distances,
                 vector[int] demands,
                 int capacity):
        self.solver = Esprc(num_customers,
                            distances,
                            demands,
                            capacity)
    
    def solve(self, vector[double] duals, int k):
        self.solver.solve(duals, k)

    def get_solution(self):
        cdef vector[vector[int]] routes
        cdef vector[double] route_costs
        
        self.solver.get_solution(routes, route_costs)
        routes_ = list(map(tuple, routes))

        return (routes_, route_costs)