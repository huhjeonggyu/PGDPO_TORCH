# user_pgdpo_base.py for SIR epidemic control
import torch, torch.nn as nn

regions = 2
DIM_X = 3*regions
DIM_Y = 0
DIM_U = regions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 2025

T = 1.0
m = 64

r = 0.0
gamma = 1.0
u_cap = 0.5
lb_X  = 0.0

epochs = 200
batch_size = 512
lr = 1e-3
N_eval_states = 1024
CRN_SEED_EU = 13579
Y0_range = None

beta = torch.tensor([0.8, 0.6], device=device)
gamma_I = torch.tensor([0.3, 0.25], device=device)
B_cost = 0.1 * torch.ones(regions, device=device)

def make_generator(seed_val: int | None):
    g = torch.Generator(device=device)
    if seed_val is not None: g.manual_seed(int(seed_val))
    return g

def sample_TmT(B: int, *, rng=None):
    return torch.rand((B,1), device=device, generator=rng).clamp_min(1e-6)*T

def sample_initial_states(B: int, *, rng=None):
    S0 = 0.95*torch.ones(B,regions, device=device)
    I0 = 0.05*torch.rand(B,regions, device=device, generator=rng)
    R0 = torch.zeros(B,regions, device=device)
    X0 = torch.cat([S0,I0,R0], dim=1)
    TmT0 = sample_TmT(B, rng=rng)
    return {"X": X0, "TmT": TmT0}, None

class DirectPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM_X+1, 128), nn.SiLU(),
            nn.Linear(128,128), nn.SiLU(),
            nn.Linear(128,DIM_U)
        )
    def forward(self, **states_dict):
        X = states_dict["X"]; TmT = states_dict["TmT"]
        u = self.net(torch.cat([X,TmT],dim=1))
        return torch.clamp(u,0.0,u_cap)

@torch.no_grad()
def simulate(policy, B:int, *, initial_states_dict=None, random_draws=None, m_steps=None, train=False):
    if initial_states_dict is None:
        initial_states_dict,_=sample_initial_states(B,rng=None)
    X = initial_states_dict["X"].clone()
    TmT0=initial_states_dict["TmT"].clone()
    steps=int(m_steps or m); dt=(TmT0/steps).view(-1,1)
    U_acc=torch.zeros((B,1),device=device)
    for i in range(steps):
        tau=TmT0-i*dt
        u=policy(X=X,TmT=tau)
        S=X[:,0:regions]; I=X[:,regions:2*regions]; R=X[:,2*regions:3*regions]
        cost = I.sum(dim=1,keepdim=True) + 0.5*((u**2)@B_cost.view(-1,1))
        U_acc -= cost*dt
        dS = -beta* S*I - u*S
        dI = beta* S*I - gamma_I*I
        dR = gamma_I*I + u*S
        X = torch.cat([S+dS*dt, I+dI*dt, R+dR*dt], dim=1).clamp(lb_X,1.0)
    return U_acc

def build_closed_form_policy():
    return None,None
