import numpy as np
import torch
from torch.autograd import Function
from gurobipy import Model, GRB, quicksum

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

def QPFunction():
    class QPFunctionFn(Function):
        @staticmethod
        def forward(ctx, Q, p, G, h, z_sum):
            ctx.Q = Q.detach().cpu().numpy()
            ctx.p = p.detach().cpu().numpy()
            ctx.G = G
            ctx.h = h
            
            ctx.nineq = ctx.G.shape[0]
            ctx.nz = ctx.G.shape[1]

            G0 = ctx.G[:-ctx.nz, :]
            h0 = ctx.h[:-ctx.nz]  

            m = Model()
            m.params.OutputFlag = 0
            m.params.Method = 0
            
            z = m.addMVar(ctx.nz, lb=0.0, vtype=GRB.CONTINUOUS)
            cineq = m.addConstr(G0 @ z <= h0)

            EPSILON = 1e-6
            m.addConstr(np.ones(ctx.nz) @ z >= (1-EPSILON) * z_sum)
            
            m.setObjective(z @ (0.5 * ctx.Q) @ z + ctx.p @ z)
            m.optimize()
            
            try:
                ctx.z_star = z.x
            except:
                print('exception in retrieving z.x')
                out = dict(Q=Q, p=p, G=G, h=h, z_sum=z_sum)
                torch.save(out, 'exception.dat')
                import sys
                sys.exit(1)
            ctx.l_star = np.concatenate((-cineq.Pi, z.RC)) 

            return torch.from_numpy(ctx.z_star).float().to(device)

        @staticmethod
        def backward(ctx, grad_output):
            if grad_output is None:
                return None, None, None, None, None

            m = np.bmat([
                [ctx.Q,                ctx.G.T @ np.diag(ctx.l_star),       np.ones((ctx.nz, 1))    ],
                [ctx.G,                np.diag(ctx.G @ ctx.z_star - ctx.h), np.zeros((ctx.nineq, 1))],
                [np.ones((1, ctx.nz)), np.zeros((1, ctx.nineq)),            np.zeros((1, 1))        ]
            ])
            
            d = grad_output.detach().cpu().numpy()            
            d = np.pad(d, (0, ctx.nineq+1))                

            try:
                pg, _, _, _ = np.linalg.lstsq(m, d, rcond=None)
            except:
                try:
                    import scipy.linalg
                    pg, *_ = scipy.linalg.lstsq(m, d)
                except:
                    out = dict(Q=ctx.Q, p=ctx.p,
                    G=ctx.G, h=ctx.h,
                    z_star=ctx.z_star, l_star=ctx.l_star,
                    grad_output=grad_output, m=m, d=d)
                    torch.save(out, 'exception.dat')
                    print('exception in lstsq', flush=True)
                    import sys
                    sys.exit(1)
                
            dz = -pg[:ctx.nz][np.newaxis,:]
            ctx.z_star = ctx.z_star[np.newaxis,:]
            
            dQ = 0.5 * (dz.T @ ctx.z_star + ctx.z_star.T @ dz)
            return torch.from_numpy(dQ).float().to(device), torch.from_numpy(dz).float().to(device), None, None, None
    return QPFunctionFn.apply

