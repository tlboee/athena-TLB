//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu>
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file g2_ambient.cpp
//  \brief Problem generator: hot ambient medium around Sgr A* (Burkert et al. 2012)
//
//  rho(r) = rho0 * (r0 / r)
//  T(r)   = T0   * (r0 / r)
//  P(r)   = rho * k_B * T / (mu * m_H)   ~ 1/r^2
//
//  Use with spherical-polar coordinates.
//========================================================================================

// C++ headers
#include <cmath>
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"

// ---------------------------------------------------------------------------
// Physical constants (CGS)
// ---------------------------------------------------------------------------
namespace {
  constexpr Real k_B   = 1.380649e-16;
  constexpr Real m_H   = 1.6726e-24;
  constexpr Real G_N   = 6.674e-8;
  constexpr Real M_sun = 1.989e33;

  Real rho0_g, T0_g, r0_g, mu_g;
  Real M_grav_g;
}

// Forward declaration (8-parameter SrcTermFunc signature)
void BHGravity(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim,
               const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc,
               AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar);

//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  rho0_g = pin->GetOrAddReal("problem", "rho0",  9.5e-22);
  T0_g   = pin->GetOrAddReal("problem", "T0",    2.0e8);
  r0_g   = pin->GetOrAddReal("problem", "r0",    1.0e16);
  mu_g   = pin->GetOrAddReal("problem", "mu",    0.5);

  Real M_BH   = pin->GetOrAddReal("problem", "M_BH",  4.31e6) * M_sun;
  int use_nsc = pin->GetOrAddInteger("problem", "use_nsc", 0);
  Real M_NSC  = 0.0;
  if (use_nsc) {
    M_NSC = pin->GetOrAddReal("problem", "M_NSC", 0.66e6) * M_sun;
    std::cout << "[g2_ambient] NSC mass: " << M_NSC/M_sun << " M_sun\n";
  }
  M_grav_g = M_BH + M_NSC;

  std::cout << "[g2_ambient] rho0   = " << rho0_g << " g/cm^3\n"
            << "[g2_ambient] T0     = " << T0_g   << " K\n"
            << "[g2_ambient] r0     = " << r0_g   << " cm\n"
            << "[g2_ambient] M_grav = " << M_grav_g/M_sun << " M_sun\n";

  EnrollUserExplicitSourceFunction(BHGravity);
}

//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Get adiabatic index from EOS
  Real gamma = peos->GetGamma();
  Real gm1   = gamma - 1.0;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {

        Real r   = pcoord->x1v(i);            // radius of cell centre [cm]
        Real rho = rho0_g * (r0_g / r);       // density [g/cm^3]
        Real T   = T0_g   * (r0_g / r);       // temperature [K]
        Real P   = rho * k_B * T / (mu_g * m_H); // pressure [dyn/cm^2]

        // ---  Primitive variables  ---
        phydro->w(IDN, k, j, i) = rho;
        phydro->w(IVX, k, j, i) = 0.0;   // v_r   = 0
        phydro->w(IVY, k, j, i) = 0.0;   // v_theta = 0
        phydro->w(IVZ, k, j, i) = 0.0;   // v_phi  = 0
        phydro->w(IPR, k, j, i) = P;

        // ---  Conserved variables  (set directly, standard Athena++ pgen pattern) ---
        // For v=0: total energy = internal energy only
        phydro->u(IDN, k, j, i) = rho;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        phydro->u(IEN, k, j, i) = P / gm1;  // e_int = P / (gamma - 1)
      }
    }
  }

  // Quick sanity check: print first and last cell
  std::cout << "[g2_ambient] Cell i=is:  r=" << pcoord->x1v(is)
            << "  rho=" << phydro->w(IDN,ks,js,is) << "\n";
  std::cout << "[g2_ambient] Cell i=ie:  r=" << pcoord->x1v(ie)
            << "  rho=" << phydro->w(IDN,ks,js,ie) << "\n";
}

//========================================================================================
// Gravitational source term (radial BH gravity, 8-parameter signature)
//========================================================================================
void BHGravity(MeshBlock *pmb, const Real time, const Real dt,
               const AthenaArray<Real> &prim,
               const AthenaArray<Real> &prim_scalar,
               const AthenaArray<Real> &bcc,
               AthenaArray<Real> &cons,
               AthenaArray<Real> &cons_scalar) {
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real r   = pmb->pcoord->x1v(i);
        Real rho = prim(IDN, k, j, i);
        Real v_r = prim(IVX, k, j, i);

        Real a_r = -G_N * M_grav_g / (r * r);

        cons(IM1, k, j, i) += dt * rho * a_r;
        cons(IEN, k, j, i) += dt * rho * v_r * a_r;
      }
    }
  }
}
