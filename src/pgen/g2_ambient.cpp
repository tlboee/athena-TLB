//========================================================================================
// Athena++ astrophysical MHD code
//! \file g2_ambient.cpp
//  Burkert+12 hot ambient medium — M3 version with passive tracer + G2 cloud
//
//  Tracer convention (Burkert+12):  tr = 0  pristine atmosphere
//                                   tr = 1  G2 cloud material
//
//  Relaxation is conditioned on the tracer: only pristine cells (tr < tr_thresh)
//  are driven back toward the Burkert equilibrium profile.
//  Default tr_thresh = 1e-4 (Burkert+12): relax only truly uncontaminated background.
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
#if NSCALARS > 0
#include "../scalars/scalars.hpp"
#endif

namespace {
  constexpr Real k_B  = 1.380649e-16;  // erg/K
  constexpr Real m_H  = 1.6726e-24;    // g
  constexpr Real G_N  = 6.674e-8;      // cgs
  constexpr Real M_sun= 1.989e33;      // g

  // Ambient medium parameters
  Real rho0_g, T0_g, r0_g, mu_g;
  Real M_grav_g;
  Real tau_relax_g;
  Real gamma_g;

  Real M_BH_g;       // BH mass (for Keplerian velocity, excludes NSC) [g]

  // Tracer threshold: tr < tr_thresh = pristine gas (apply relaxation)
  Real tr_thresh_g;

  // G2 cloud (M3)
  Real r_G2_g;      // cloud center distance from BH [cm]
  Real th_G2_g;     // cloud center polar angle [rad]
  Real R_cloud_g;   // cloud radius [cm]
  Real rho_G2_g;    // cloud density [g/cm3] (pressure equilibrium)
  Real T_G2_g;      // cloud temperature [K]
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

  Real M_BH   = pin->GetOrAddReal("problem", "M_BH",  4.31e6) * M_sun;
  M_BH_g = M_BH;   // save for ProblemGenerator Keplerian velocity
  int  use_nsc = pin->GetOrAddInteger("problem", "use_nsc", 0);
  Real M_NSC  = 0.0;
  if (use_nsc) M_NSC = pin->GetOrAddReal("problem", "M_NSC", 0.66e6) * M_sun;
  M_grav_g = M_BH + M_NSC;

  tau_relax_g = pin->GetOrAddReal("problem", "tau_relax", 0.0);
  tr_thresh_g = pin->GetOrAddReal("problem", "tr_thresh",  1e-4);

  // G2 cloud position from Burkert+12 CC01 (Cartesian, orbital plane)
  Real x_G2 = pin->GetOrAddReal("problem", "x_G2", -7.22e16);
  Real z_G2 = pin->GetOrAddReal("problem", "z_G2",  2.21e16);
  r_G2_g    = std::sqrt(x_G2*x_G2 + z_G2*z_G2);
  th_G2_g   = pin->GetOrAddReal("problem", "th_G2", std::acos(z_G2/r_G2_g));
  R_cloud_g = pin->GetOrAddReal("problem", "R_cloud", 1.87e15);
  T_G2_g    = pin->GetOrAddReal("problem", "T_G2",    1.0e4);

  // Cloud density from pressure equilibrium with ambient at cloud center
  Real rho_hot_c = rho0_g * (r0_g / r_G2_g);
  Real T_hot_c   = T0_g   * (r0_g / r_G2_g);
  rho_G2_g = rho_hot_c * (T_hot_c / T_G2_g);

  Real M_eq = 2.0 * k_B * T0_g * r0_g / (G_N * mu_g * m_H);
  std::cout << "[g2_ambient] M_grav    = " << M_grav_g/M_sun << " M_sun\n";
  std::cout << "[g2_ambient] M_eq      = " << M_eq/M_sun     << " M_sun\n";
  std::cout << "[g2_ambient] M/M_eq    = " << M_grav_g/M_eq  << "\n";
  std::cout << "[g2_ambient] tau_relax = " << tau_relax_g/3.156e7 << " yr\n";
  std::cout << "[g2_ambient] G2 center: r=" << r_G2_g/1e16 << "e16 cm"
            << "  th=" << th_G2_g << " rad\n";
  std::cout << "[g2_ambient] G2 cloud:  R=" << R_cloud_g/1e15 << "e15 cm"
            << "  rho=" << rho_G2_g << " g/cm3"
            << "  T=" << T_G2_g << " K\n";
#if NSCALARS == 0
  std::cout << "[g2_ambient] WARNING: compiled without NSCALARS -- "
               "tracer and G2 cloud disabled!\n";
#endif

  EnrollUserExplicitSourceFunction(BHGravityAndRelax);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, BurkertInnerBC);
}

//----------------------------------------------------------------------------------------
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gamma = peos->GetGamma(), gm1 = gamma - 1.0;

  // Cloud center in Cartesian (in the 2D simulation plane: x=r*sin(th), z=r*cos(th))
  Real xc = r_G2_g * std::sin(th_G2_g);
  Real zc = r_G2_g * std::cos(th_G2_g);

  for (int k=ks; k<=ke; ++k)
  for (int j=js; j<=je; ++j)
  for (int i=is; i<=ie; ++i) {
    Real r  = pcoord->x1v(i);
    Real th = pcoord->x2v(j);

    // Burkert ambient profile
    Real rho_amb = rho0_g * (r0_g / r);
    Real T_amb   = T0_g   * (r0_g / r);
    Real P_amb   = rho_amb * k_B * T_amb / (mu_g * m_H);

    // Distance from G2 cloud center (via Cartesian conversion)
    Real xi = r * std::sin(th);
    Real zi = r * std::cos(th);
    Real d  = std::sqrt((xi-xc)*(xi-xc) + (zi-zc)*(zi-zc));

    bool in_cloud = (d < R_cloud_g);

    Real rho = in_cloud ? rho_G2_g : rho_amb;
    Real P   = in_cloud ? rho_G2_g * k_B * T_G2_g / (mu_g * m_H) : P_amb;

    // Keplerian tangential velocity for cloud cells (circular orbit at r_G2).
    // v_th < 0: cloud moves toward smaller theta (prograde, approaching north pole).
    // For G2 approaching periapsis from theta~73 deg, prograde = decreasing theta.
    Real v_th = 0.0;
    if (in_cloud) {
      Real v_kep = std::sqrt(G_N * M_BH_g / r_G2_g);  // circular speed at r_G2
      v_th = -v_kep;   // prograde: decreasing theta; flip sign for retrograde
    }

    phydro->w(IDN,k,j,i) = rho;
    phydro->w(IVX,k,j,i) = 0.0;
    phydro->w(IVY,k,j,i) = v_th;
    phydro->w(IVZ,k,j,i) = 0.0;
    phydro->w(IPR,k,j,i) = P;

    phydro->u(IDN,k,j,i) = rho;
    phydro->u(IM1,k,j,i) = 0.0;
    phydro->u(IM2,k,j,i) = rho * v_th;
    phydro->u(IM3,k,j,i) = 0.0;
    phydro->u(IEN,k,j,i) = P / gm1 + 0.5 * rho * v_th * v_th;

#if NSCALARS > 0
    // tr=0: pristine atmosphere,  tr=1: G2 cloud material  (Burkert+12 convention)
    Real tr = in_cloud ? 1.0 : 0.0;
    pscalars->r(0, k, j, i) = tr;
    pscalars->s(0, k, j, i) = rho * tr;  // conserved: rho * tracer
#endif
  }
}

//----------------------------------------------------------------------------------------
// Inner boundary: Dirichlet Burkert profile (inflow only)
void BurkertInnerBC(MeshBlock *pmb, Coordinates *pco,
                    AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j) {
    Real vr_active = prim(IVX, k, j, il);
    for (int i=1; i<=ngh; ++i) {
      int ig = il - i;
      Real r   = pco->x1v(ig);
      Real rho = rho0_g * (r0_g / r);
      Real T   = T0_g   * (r0_g / r);
      Real P   = rho * k_B * T / (mu_g * m_H);
      prim(IDN, k, j, ig) = rho;
      prim(IPR, k, j, ig) = P;
      prim(IVX, k, j, ig) = std::min(vr_active, 0.0);
      prim(IVY, k, j, ig) = 0.0;
      prim(IVZ, k, j, ig) = 0.0;
    }
  }
  // Scalar BC: default outflow (copy from innermost active cell).
  // G2 starts far from inner boundary, so inner tracer stays ~0 (pristine).
}

//----------------------------------------------------------------------------------------
// Source term: gravity + tracer-conditioned relaxation
void BHGravityAndRelax(MeshBlock *pmb, const Real time, const Real dt,
                       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                       AthenaArray<Real> &cons_scalar) {
  const Real gm1       = gamma_g - 1.0;
  const bool do_instant = (tau_relax_g == 0.0);   // direct Dirichlet reset each step
  const bool do_relax   = (tau_relax_g >  0.0);   // exponential nudge with timescale tau

  for (int k=pmb->ks; k<=pmb->ke; ++k)
  for (int j=pmb->js; j<=pmb->je; ++j)
  for (int i=pmb->is; i<=pmb->ie; ++i) {
    Real r   = pmb->pcoord->x1v(i);
    Real rho = prim(IDN, k, j, i);
    Real v_r = prim(IVX, k, j, i);
    Real v_t = prim(IVY, k, j, i);
    Real P   = prim(IPR, k, j, i);

    // 1. Gravity (always)
    Real a_r = -G_N * M_grav_g / (r * r);
    cons(IM1, k, j, i) += dt * rho * a_r;
    cons(IEN, k, j, i) += dt * rho * v_r * a_r;

    // 2. Tracer-conditioned Burkert relaxation
    //    tr < tr_thresh → pristine background → apply relaxation
    //    tr >= tr_thresh → cloud material     → evolve freely
#if NSCALARS > 0
    const Real tr      = prim_scalar(0, k, j, i);   // primitive tracer (= s/rho)
    const bool pristine = (tr < tr_thresh_g);
#else
    const Real tr       = 0.0;
    const bool pristine = true;  // no tracer: always relax (M2 fallback)
#endif

    if (do_instant && pristine) {
      // tau_relax = 0: hard reset to Burkert equilibrium every timestep
      Real rho_eq = rho0_g * (r0_g / r);
      Real P_eq   = rho_eq * k_B * T0_g * (r0_g / r) / (mu_g * m_H);
      cons(IDN, k, j, i) = rho_eq;
      cons(IM1, k, j, i) = 0.0;
      cons(IM2, k, j, i) = 0.0;
      cons(IEN, k, j, i) = P_eq / gm1;
#if NSCALARS > 0
      cons_scalar(0, k, j, i) = rho_eq * tr;   // preserve tracer fraction
#endif
    } else if (do_relax && pristine) {
      // tau_relax > 0: exponential nudge toward equilibrium
      Real rho_eq  = rho0_g * (r0_g / r);
      Real P_eq    = rho_eq * k_B * T0_g * (r0_g / r) / (mu_g * m_H);
      Real inv_tau = dt / tau_relax_g;
      cons(IDN, k, j, i) -= inv_tau * (rho - rho_eq);
      cons(IM1, k, j, i) -= inv_tau * rho * v_r;
      cons(IM2, k, j, i) -= inv_tau * rho * v_t;
      cons(IEN, k, j, i) -= inv_tau * (P - P_eq) / gm1;
#if NSCALARS > 0
      cons_scalar(0, k, j, i) = cons(IDN, k, j, i) * tr;  // keep tr = s/rho consistent
#endif
    }
    // cloud cells (pristine=false): no relaxation → cloud evolves freely
  }
}
