"""Pure-numpy mirror of the SFAgent tensor ops to validate shapes / einsum /
gather indexing WITHOUT torch. Any mismatch here would be a real bug."""
import os as _os, sys as _sys
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _PKG)
import numpy as np
B,N,S,A,K,H = 4,6,8,5,5,16
rng=np.random.RandomState(0)

# ---- SFQNet.forward -> psi [B,N,A,K]; q_values = einsum psi,w ----
psi = rng.randn(B,N,A,K)
w1 = rng.randn(K)                       # shared w
q_shared = np.einsum('bnak,k->bna', psi, w1)
assert q_shared.shape==(B,N,A), q_shared.shape
w_bn = rng.randn(B,N,K)                 # per-agent w
q_bn = np.einsum('bnak,bnk->bna', psi, w_bn)
assert q_bn.shape==(B,N,A), q_bn.shape

# ---- USFA SF-TD: gather psi at taken action A -> psi_taken [B,N,K] ----
Aidx = rng.randint(0,A,size=(B,N))
# torch: psi.gather(2, A.view(B,N,1,1).expand(-1,-1,1,K)).squeeze(2)
psi_taken = np.take_along_axis(psi, Aidx[:,:,None,None], axis=2).squeeze(2)
assert psi_taken.shape==(B,N,K), psi_taken.shape
# manual check one element
b,n=2,3
assert np.allclose(psi_taken[b,n], psi[b,n,Aidx[b,n]]), "gather mismatch"

# next: a2 = argmax_a psi2.w ; psi2_sel [B,N,K]
w_t = rng.randn(N,K)
psi2 = rng.randn(B,N,A,K)
q2 = np.einsum('bnak,nk->bna', psi2, w_t)
a2 = q2.argmax(-1)
psi2_sel = np.take_along_axis(psi2, a2[:,:,None,None], axis=2).squeeze(2)
assert psi2_sel.shape==(B,N,K)
PHI2 = rng.randn(B,N,K); D=rng.randint(0,2,size=(B,1,1)).astype(float); gamma=0.99
target = PHI2 + gamma*(1.0-D)*psi2_sel
assert target.shape==(B,N,K), target.shape

# ---- composer per-agent TD: r = phi2 . w  [B,N] ----
r = np.einsum('bnk,nk->bn', PHI2, w_t)
assert r.shape==(B,N), r.shape
u = rng.randn(B,N,A)
u_taken = np.take_along_axis(u, Aidx[:,:,None], axis=-1).squeeze(-1)
assert u_taken.shape==(B,N)
u2max = rng.randn(B,N,A).max(-1)
D2=rng.randint(0,2,size=(B,1)).astype(float)
tgt = r + gamma*(1.0-D2)*u2max
assert tgt.shape==(B,N), tgt.shape
# verify r element
b,n=1,4
assert np.allclose(r[b,n], PHI2[b,n]@w_t[n]), "reward dot mismatch"

# ---- hetero task matrix + per-ts reward map ----
W_hetero = rng.randn(N,K)                        # sample_hetero_task -> [N,K]
assert W_hetero.shape==(N,K)
# per-ts reward r_i = phi_i . w_i  (matches make_sf_reward w_map path)
phi_step = rng.randn(N,K)
r_map = np.einsum('nk,nk->n', phi_step, W_hetero)
assert r_map.shape==(N,), r_map.shape
n=2
assert np.allclose(r_map[n], phi_step[n]@W_hetero[n]), "hetero reward mismatch"

print("ALL SHAPE/INDEX CHECKS PASSED")
