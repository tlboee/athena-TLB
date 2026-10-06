//========================================================================================
// Athena++ astrophysical MHD code
//! \file g2_ambient.cpp
//  Burkert+12 hot ambient medium; inner Dirichlet BC + optional thermal relaxation
//========================================================================================
#include <algorithm>
#include <cmath>
#include <iostream>
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../bvals/bvals.hpp"

namespace {
  constexpr Real k_B  = 1.380649e-16;  // erg/K
  constexpr Real m_H  = 1.6726e-24;    // g
  constexpr Real G_N  = 6.674e-8;      // cgs
  constexpr Real M_sun= 1.989e33;      // g
  Real rho0_g, T0_g, r0_g, mu_g;
  Real M_grav_g;
  Real tau_relax_g;   // relaxation timescale [s]; 0 = disabled
  Real gamma_g;       // cached adiabatic index
}

// Forward declarations
void BHGravityAndRelax(MeshBlock *pmb, const Real time, const Real dt,
                       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                       AthenaArray<Real> &cons_scalar);

void BurkertInnerBC(MeshBlock *pmb, Coordinates *pco,
                    AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//----------------------------------------------------------------------------------------
void Mesh::InitUserMeshData(ParameterInput *pin) {
  rho0_g = pin->GetOrAddReal("problem", "rho0",  9.5e-22);
  T0_g   = pin->GetOrAddReal("problem", "T0",    2.0e8);
  r0_g   = pin->GetOrAddReal("problem", "r0",    1.0e16);
  mu_g   = pin->GetOrAddReal("problem", "mu",    0.5);
  gamma_g= pin->GetOrAddReal("hydro",   "gamma", 1.6666666667);

  Real M_BH  = pin->GetOrAddReal("problem", "M_BH",  4.31e6) * M_sun;
  int  use_nsc = pin->GetOrAddInteger("problem", "use_nsc", 0);
  Real M_NSC = 0.0;
  if (use_nsc) M_NSC = pin->GetOrAddReal("problem", "M_NSC", 0.66e6) * M_sun;
  M_grav_g = M_BH + M_NSC;

  tau_relax_g = pin->GetOrAddReal("problem", "tau_relax", 0.0);

  // Diagnostic: hydrostatic equilibrium mass
  Real M_eq = 2.0 * k_B * T0_g * r0_g / (G_N * mu_g * m_H);
  std::cout << "[g2_ambient] M_BH   = " << M_BH/M_sun    << " M_sun\n";
  std::cout << "[g2_ambient] M_NSC  = " << M_NSC/M_sun   << " M_sun\n";
  std::cout << "[g2_ambient] M_grav = " << M_grav_g/M_sun<< " M_sun\n";
  std::cout << "[g2_ambient] M_eq   = " << M_eq/M_sun    << " M_sun\n";
  std::cout << "[g2_ambient] M/M_eq = " << M_grav_g/M_eq << "\n";
  std::cout << "[g2_ambient] tau_relax = " << tau_relax_g << " s  ("
            << tau_relax_g/3.156e7 << " yr)\n";

  EnrollUserExplicitSourceFunction(BHGravityAndRelax);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, BurkertInnerBC);
}

//----------------------------------------------------------------------------------------
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gamma = peos->GetGamma(), gm1 = gamma - 1.0;
  for (int k=ks; k<=ke; ++k)
  for (int j=js; j<=je; ++j)
  for (int i=is; i<=ie; ++i) {
    Real r   = pcoord->x1v(i);
    Real rho = rho0_g * (r0_g / r);
    Real T   = T0_g   * (r0_g / r);
    Real P   = rho * k_B * T / (mu_g * m_H);
    phydro->w(IDN,k,j,i) = rho;
    phydro->w(IVX,k,j,i) = 0.0;
    phydro->w(IVY,k,j,i) = 0.0;
    phydro->w(IVZ,k,j,i) = 0.0;
    phydro->w(IPR,k,j,i) = P;
    phydro->u(IDN,k,j,i) = rho;
    phydro->u(IM1,k,j,i) = 0.0;
    phydro->u(IM2,k,j,i) = 0.0;
    phydro->u(IM3,k,j,i) = 0.0;
    phydro->u(IEN,k,j,i) = P / gm1;
  }
}

//----------------------------------------------------------------------------------------
// Inner boundary: Dirichlet Burkert profile (inflow forbidden)
void BurkertInnerBC(MeshBlock *pmb, Coordinates *pco,
                    AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j) {
    Real vr_active = prim(IVX, k, j, il);  // first active cell
    for (int i=1; i<=ngh; ++i) {
      int ig = il - i;
      Real r   = pco->x1v(ig);
      Real rho = rho0_g * (r0_g / r);
      Real T   = T0_g   * (r0_g / r);
      Real P   = rho * k_B * T / (mu_g * m_H);
      prim(IDN, k, j, ig) = rho;
      prim(IPR, k, j, ig) = P;
      prim(IVX, k, j, ig) = std::min(vr_active, 0.0);  // allow inflow only
      prim(IVY, k, j, ig) = 0.0;
      prim(IVZ, k, j, ig) = 0.0;
    }
  }
}

//----------------------------------------------------------------------------------------
// Combined source term: BH gravity  +  optional Burkert thermal relaxation
// tau_relax > 0  -->  drives (rho, v, P)  -->  (rho_eq, 0, P_eq) on timescale tau_relax
// For M3: replace relaxation with a tracer-conditioned reset (pristine cells only)
void BHGravityAndRelax(MeshBlock *pmb, const Real time, const Real dt,
                       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                       AthenaArray<Real> &cons_scalar) {
  const Real gm1      = gamma_g - 1.0;
  const bool do_relax = (tau_relax_g > 0.0);

  for (int k=pmb->ks; k<=pmb->ke; ++k)
  for (int j=pmb->js; j<=pmb->je; ++j)
  for (int i=pmb->is; i<=pmb->ie; ++i) {
    Real r   = pmb->pcoord->x1v(i);
    Real rho = prim(IDN, k, j, i);
    Real v_r = prim(IVX, k, j, i);
    Real v_t = prim(IVY, k, j, i);
    Real P   = prim(IPR, k, j, i);

    // 1. BH (+ NSC) point-mass gravity
    Real a_r = -G_N * M_grav_g / (r * r);
    cons(IM1, k, j, i) += dt * rho * a_r;
    cons(IEN, k, j, i) += dt * rho * v_r * a_r;

    // 2. Burkert relaxation  (Burkert+12 eq. trick: keep ambient pristine)
    if (do_relax) {
      Real rho_eq = rho0_g * (r0_g / r);
      Real P_eq   = rho_eq * k_B * T0_g * (r0_g / r) / (mu_g * m_H);
      Real inv_tau = dt / tau_relax_g;
      // drive conserved variables toward equilibrium
      cons(IDN, k, j, i) -= inv_tau * (rho - rho_eq);
      cons(IM1, k, j, i) -= inv_tau * rho * v_r;
      cons(IM2, k, j, i) -= inv_tau * rho * v_t;
      cons(IEN, k, j, i) -= inv_tau * (P - P_eq) / gm1;
    }
  }
}