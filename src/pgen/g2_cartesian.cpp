//========================================================================================
// G2-Wolke im heissen Medium um Sgr A*, 2D kartesisch in der Bahnebene
// Setup nach Burkert et al. (2012) / Schartmann et al. (2012), Modell CC01
//
// Konfiguration:
//   python3 configure.py --prob=g2_cartesian --coord=cartesian --flux=hllc \
//                        --nscalars=1 --nghost=4 -mpi [-hdf5 --hdf5_path=...]
//
// Physik:
//   - Schwarzes Loch (Punktmasse) im Ursprung, Gravitation als Quellterm.
//     Energieterm mit mittlerem Impuls -> Gravitation aendert die innere Energie nicht.
//   - Atmosphaere: rho = rho0 r0/r, T = T0 r0/r, hydrostatisch fuer
//     T0 = mu m_H G M / (2 k_B r0)  (wird berechnet, wenn T0 <= 0 im Input)
//   - Bereich r < r_in um das Schwarze Loch: jeder Zeitschritt auf
//     Gleichgewicht zurueckgesetzt (Senke, haelt den Zeitschritt endlich)
//   - Tracer r0: 1 = Wolkenmaterial, 0 = unberuehrtes Umgebungsgas
//   - Zellen mit tr < tr_thresh werden jeden Schritt auf Gleichgewicht gesetzt
//     (Burkert+12: tr <= 1e-4). tr wird aus dem AKTUELLEN Zustand berechnet.
//   - Temperaturboden T_floor (Default 1e4 K) fuer nicht-pristine Zellen.
//   - Raender: outflow plus Pufferzone (edge_buffer Zellen), die jeden Schritt auf
//     Gleichgewicht mit v = 0 und tr = 0 gesetzt wird (Ausfluss ja, Einstroem nein).
//
// Diagnostik (History-Ausgabe, kumulativ, Einheit: Tracer-Masse = sum rho*tr*dV):
//   M_sink   : in der Senke entferntes Wolkengas
//   M_buffer : in der Pufferzone am Rand entferntes Wolkengas
//   M_reset  : Aenderung durch das Zuruecksetzen pristiner Zellen (vorzeichenbehaftet)
//   M_buf_tr>1/3, M_buf_tr>1/2 : Anteil von M_buffer aus Zellen mit Tracer > 1/3 bzw. > 1/2
//   Die Zaehler summieren ueber alle RK-Stufen und ueberzaehlen daher um einen fuer alle
//   Kanaele gleichen Faktor; die exakte Gesamtabnahme steht in der Spalte "0-scalar".
//   Aufteilung: Anteil eines Kanals = Zaehler / Summe der Zaehler.
//========================================================================================

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

namespace {

// Naturkonstanten (cgs)
constexpr Real G_N   = 6.674e-8;
constexpr Real k_B   = 1.380649e-16;
constexpr Real m_H   = 1.6726e-24;
constexpr Real M_sun = 1.989e33;
constexpr Real km    = 1.0e5;

// Indizes der Zaehler in ruser_meshblock_data[0]
enum {CNT_SINK = 0, CNT_BUFFER = 1, CNT_RESET = 2, CNT_BUF_THIRD = 3, CNT_BUF_HALF = 4,
      N_CNT = 5};

// Globale Parameter (in InitUserMeshData gesetzt)
Real GM_g, rho0_g, r0_g, T0_g, mu_g, r_in_g, tr_thresh_g, T_floor_g;
Real x1min_g, x1max_g, x2min_g, x2max_g, buf_g;

// Gleichgewichtsprofil; innerhalb r_in konstant (ausgeschnittener Bereich)
void Equilibrium(Real r, Real &rho, Real &P) {
  const Real rr = std::max(r, r_in_g);
  rho = rho0_g * r0_g / rr;
  const Real T = T0_g * r0_g / rr;
  P = rho * k_B * T / (mu_g * m_H);
}

// Zelle auf Gleichgewicht mit v = 0 setzen; Tracer-Konzentration tr bleibt erhalten
void ResetCell(AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar,
               int k, int j, int i, Real r, Real gm1, Real tr) {
  Real rho_eq, P_eq;
  Equilibrium(r, rho_eq, P_eq);
  cons(IDN, k, j, i) = rho_eq;
  cons(IM1, k, j, i) = 0.0;
  cons(IM2, k, j, i) = 0.0;
  cons(IM3, k, j, i) = 0.0;
  cons(IEN, k, j, i) = P_eq / gm1;
  cons_scalar(0, k, j, i) = rho_eq * tr;
}

void GravityAndRelax(MeshBlock *pmb, const Real time, const Real dt,
                     const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                     const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                     AthenaArray<Real> &cons_scalar);

// History-Funktionen: Zaehler des MeshBlocks (Athena++ summiert ueber alle Bloecke)
Real HistSink(MeshBlock *pmb, int iout)   { return pmb->ruser_meshblock_data[0](CNT_SINK); }
Real HistBuffer(MeshBlock *pmb, int iout) { return pmb->ruser_meshblock_data[0](CNT_BUFFER); }
Real HistReset(MeshBlock *pmb, int iout)  { return pmb->ruser_meshblock_data[0](CNT_RESET); }
Real HistBufThird(MeshBlock *pmb, int iout) { return pmb->ruser_meshblock_data[0](CNT_BUF_THIRD); }
Real HistBufHalf(MeshBlock *pmb, int iout)  { return pmb->ruser_meshblock_data[0](CNT_BUF_HALF); }

}  // namespace

//----------------------------------------------------------------------------------------
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in g2_cartesian.cpp: requires --coord=cartesian" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
#if NSCALARS < 1 || !NON_BAROTROPIC_EOS
  std::stringstream msg;
  msg << "### FATAL ERROR in g2_cartesian.cpp: requires --nscalars>=1 and adiabatic EOS"
      << std::endl;
  throw std::runtime_error(msg.str().c_str());
#endif

  GM_g        = G_N * pin->GetOrAddReal("problem", "M_BH", 4.31e6) * M_sun;
  rho0_g      = pin->GetOrAddReal("problem", "rho0", 9.5e-22);
  r0_g        = pin->GetOrAddReal("problem", "r0", 1.0e16);
  mu_g        = pin->GetOrAddReal("problem", "mu", 0.614);
  T0_g        = pin->GetOrAddReal("problem", "T0", -1.0);
  r_in_g      = pin->GetOrAddReal("problem", "r_in", 1.0e15);
  tr_thresh_g = pin->GetOrAddReal("problem", "tr_thresh", 1.0e-4);
  T_floor_g   = pin->GetOrAddReal("problem", "T_floor", 1.0e4);

  // T0 <= 0: exakt hydrostatisch, aus k_B T r / (mu m_H) = G M / 2
  if (T0_g <= 0.0) T0_g = mu_g * m_H * GM_g / (2.0 * k_B * r0_g);

  // Pufferzone am Rand: edge_buffer Zellen breit
  x1min_g = mesh_size.x1min;
  x1max_g = mesh_size.x1max;
  x2min_g = mesh_size.x2min;
  x2max_g = mesh_size.x2max;
  const Real dx = (x1max_g - x1min_g) / mesh_size.nx1;
  const Real dy = (x2max_g - x2min_g) / mesh_size.nx2;
  const int nbuf = pin->GetOrAddInteger("problem", "edge_buffer", 2);
  buf_g = nbuf * std::max(dx, dy);

  EnrollUserExplicitSourceFunction(GravityAndRelax);

  AllocateUserHistoryOutput(N_CNT);
  EnrollUserHistoryOutput(CNT_SINK,   HistSink,   "M_sink");
  EnrollUserHistoryOutput(CNT_BUFFER, HistBuffer, "M_buffer");
  EnrollUserHistoryOutput(CNT_RESET,  HistReset,  "M_reset");
  EnrollUserHistoryOutput(CNT_BUF_THIRD, HistBufThird, "M_buf_tr>1/3");
  EnrollUserHistoryOutput(CNT_BUF_HALF,  HistBufHalf,  "M_buf_tr>1/2");

  if (Globals::my_rank == 0) {
    std::cout << "[g2_cartesian] mu = " << mu_g << ", T0 = " << T0_g << " K"
              << ", r_in = " << r_in_g << " cm, tr_thresh = " << tr_thresh_g
              << ", edge buffer = " << buf_g << " cm, T_floor = " << T_floor_g << " K"
              << std::endl;
  }
}

//----------------------------------------------------------------------------------------
// Pro MeshBlock: Feld fuer die drei Zaehler (wird in Restart-Dateien mitgespeichert)
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(N_CNT);
  for (int n=0; n<N_CNT; ++n) ruser_meshblock_data[0](n) = 0.0;
}

//----------------------------------------------------------------------------------------
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  const Real gm1 = peos->GetGamma() - 1.0;

  const Real xc  = pin->GetOrAddReal("problem", "x_cloud", -7.22e16);
  const Real yc  = pin->GetOrAddReal("problem", "y_cloud",  2.21e16);
  const Real vxc = pin->GetOrAddReal("problem", "vx_cloud", 794.59) * km;  // Input in km/s
  const Real vyc = pin->GetOrAddReal("problem", "vy_cloud",  48.45) * km;
  const Real Rc  = pin->GetOrAddReal("problem", "R_cloud", 1.87e15);
  const Real Tc  = pin->GetOrAddReal("problem", "T_cloud", 1.0e4);
  Real rhoc      = pin->GetOrAddReal("problem", "rho_cloud", 0.0);

  // rho_cloud <= 0: Dichte aus Druckgleichgewicht am Wolkenzentrum bei T_cloud
  // rho_cloud  > 0: feste Dichte (z.B. Burkert+12 Tabelle 1: 6.21e-19)
  // In beiden Faellen gilt in der Wolke P = P_eq(r) (lokales Druckgleichgewicht).
  Real rho_a, P_a;
  Equilibrium(std::sqrt(xc*xc + yc*yc), rho_a, P_a);
  if (rhoc <= 0.0) rhoc = P_a * mu_g * m_H / (k_B * Tc);

  if (gid == 0) {
    std::cout << "[g2_cartesian] rho_cloud = " << rhoc << " g/cm^3, chi = "
              << rhoc / rho_a << ", T_cloud(eff) = "
              << P_a * mu_g * m_H / (k_B * rhoc) << " K" << std::endl;
  }

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        const Real x = pcoord->x1v(i);
        const Real y = pcoord->x2v(j);
        Real rho, P;
        Equilibrium(std::sqrt(x*x + y*y), rho, P);
        Real vx = 0.0, vy = 0.0, tr = 0.0;

        const Real dx = x - xc, dy = y - yc;
        if (dx*dx + dy*dy < Rc*Rc) {   // Wolke: P bleibt P_eq(r)
          rho = rhoc;
          vx  = vxc;
          vy  = vyc;
          tr  = 1.0;
        }

        phydro->u(IDN, k, j, i) = rho;
        phydro->u(IM1, k, j, i) = rho * vx;
        phydro->u(IM2, k, j, i) = rho * vy;
        phydro->u(IM3, k, j, i) = 0.0;
        phydro->u(IEN, k, j, i) = P / gm1 + 0.5 * rho * (vx*vx + vy*vy);
        pscalars->s(0, k, j, i) = rho * tr;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
namespace {

void GravityAndRelax(MeshBlock *pmb, const Real time, const Real dt,
                     const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                     const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                     AthenaArray<Real> &cons_scalar) {
  const Real gm1  = pmb->peos->GetGamma() - 1.0;
  const Real eint_per_rho_floor = k_B * T_floor_g / (mu_g * m_H * gm1);
  AthenaArray<Real> &cnt = pmb->ruser_meshblock_data[0];

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real x = pmb->pcoord->x1v(i);
        const Real y = pmb->pcoord->x2v(j);
        const Real r = std::sqrt(x*x + y*y);
        const Real rho = cons(IDN, k, j, i);
        const Real vol = pmb->pcoord->GetCellVolume(k, j, i);

        // 1. Senke, Pufferzone am Rand und unphysikalische Dichte: zuruecksetzen
        const bool in_sink   = (r < r_in_g);
        const bool in_buffer = (x < x1min_g + buf_g) || (x > x1max_g - buf_g) ||
                               (y < x2min_g + buf_g) || (y > x2max_g - buf_g);
        if (in_sink || in_buffer || !(rho > 0.0)) {
          const Real s_old = cons_scalar(0, k, j, i) * vol;
          if (in_sink)        cnt(CNT_SINK)   += s_old;
          else if (in_buffer) {
            cnt(CNT_BUFFER) += s_old;
            const Real trc = (rho > 0.0) ? cons_scalar(0, k, j, i) / rho : 0.0;
            if (trc > 1.0/3.0) cnt(CNT_BUF_THIRD) += s_old;
            if (trc > 0.5)     cnt(CNT_BUF_HALF)  += s_old;
          }
          else                cnt(CNT_RESET)  += s_old;
          ResetCell(cons, cons_scalar, k, j, i, r, gm1, 0.0);
          continue;
        }

        // 2. Gravitation, energieerhaltend: dE = dt a . (m_alt + m_neu)/2 = dE_kin
        const Real r3  = r * r * r;
        const Real ax  = -GM_g * x / r3;
        const Real ay  = -GM_g * y / r3;
        const Real m1  = cons(IM1, k, j, i);
        const Real m2  = cons(IM2, k, j, i);
        const Real dm1 = dt * rho * ax;
        const Real dm2 = dt * rho * ay;
        cons(IM1, k, j, i) = m1 + dm1;
        cons(IM2, k, j, i) = m2 + dm2;
        cons(IEN, k, j, i) += dt * (ax * (m1 + 0.5 * dm1) + ay * (m2 + 0.5 * dm2));

        // 3. Pristine Zellen auf Gleichgewicht setzen (Tracer aus aktuellem Zustand)
        const Real tr = cons_scalar(0, k, j, i) / rho;
        if (tr < tr_thresh_g) {
          const Real s_old = cons_scalar(0, k, j, i) * vol;
          ResetCell(cons, cons_scalar, k, j, i, r, gm1, tr);
          cnt(CNT_RESET) += s_old - cons_scalar(0, k, j, i) * vol;
          continue;
        }

        // 4. Temperaturboden fuer Wolken-/Mischgas
        const Real ekin = 0.5 * (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i))
                                 + SQR(cons(IM3, k, j, i))) / rho;
        const Real eint_min = rho * eint_per_rho_floor;
        if (cons(IEN, k, j, i) - ekin < eint_min) cons(IEN, k, j, i) = ekin + eint_min;
      }
    }
  }
}

}  // namespace
