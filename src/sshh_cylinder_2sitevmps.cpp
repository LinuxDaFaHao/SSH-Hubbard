#include "gqdouble.h"
#include "operators.h"
#include <vector>
#include "qlmps/qlmps.h"
#include "singlesiteupdate2.h"
#include "twositeupdate2.h"
#include "myutil.h"
#include "two_site_update_noised_finite_vmps_mpi_impl2.h"

using namespace qlmps;
using namespace qlten;
using namespace std;

#include "params_case.h"

int main(int argc, char *argv[]) {
  namespace mpi = boost::mpi;
  mpi::environment env;
  mpi::communicator world;
  CaseParams params(argv[1]);
  unsigned Lx = params.Lx, Ly = params.Ly, Np = params.Np;
  unsigned N = Lx * Ly + (2 * Lx * Ly - Ly) * Np;
  cout << "System size = (" << Lx << "," << Ly << ")" << endl;
  cout << "The number of electron sites =" << Lx * Ly << endl;
  cout << "The number of phonon pseudosite (per bond) =" << Np << endl;
  cout << "The number of phonon pseudosite (total) =" << (2 * Lx * Ly - Ly) * Np << endl;
  cout << "The total number of sites = " << N << endl;
  float t = params.t, g = params.g, U = params.U, omega = params.omega;
  cout << "Model parameter: t = " << t << ", g = " << g << ", U = " << U << ", omega = " << omega << endl;
  clock_t startTime, endTime;
  startTime = clock();
  OperatorInitial();
  vector<IndexT> pb_out_set(N);
  vector<long> Tx(N, -1), Ty(N, -1), ElectronSite(Lx * Ly);
  auto iter = ElectronSite.begin();
  // translation along x(for electron) and translation along y(for electron);
  for (size_t i = 0; i < N; ++i) {
    size_t residue = i % ((2 * Np + 1) * Ly);
    if (residue < (Np + 1) * Ly && residue % (Np + 1) == 0) {
      pb_out_set[i] = pb_outF;
      *iter = i;
      iter++;
    } else pb_out_set[i] = pb_outB;
  }
  SiteVec<TenElemT, U1U1QN> sites = SiteVec<TenElemT, U1U1QN>(pb_out_set);
  MPO<Tensor> mpo(N);
  for (size_t i = 0; i < mpo.size(); i++) {
    std::string filename = kMpoPath + "/" +
        kMpoTenBaseName + std::to_string(i) + "." + kQLTenFileSuffix;
    mpo.LoadTen(i, filename);
  }

  cout << "MPO loaded." << endl;
  using FiniteMPST = qlmps::FiniteMPS<TenElemT, U1U1QN>;
  FiniteMPST mps(sites);

  if (world.rank() == 0&&params.TotalThreads > 2) {
      qlten::hp_numeric::SetTensorManipulationThreads(params.TotalThreads - 2);
  } else {
    qlten::hp_numeric::SetTensorManipulationThreads(params.TotalThreads);
  }
  qlmps::FiniteVMPSSweepParams sweep_params(
      params.Sweeps,
      params.Dmin, params.Dmax, params.CutOff,
      qlmps::LanczosParams(params.LanczErr, params.MaxLanczIter),
      params.noise
  );

  std::vector<long unsigned int> stat_labs(N, 1);
  size_t sitenumber_perhole;
  if (params.Numhole > 0) {
    sitenumber_perhole = ElectronSite.size() / params.Numhole;
  } else {
    sitenumber_perhole = 4 * ElectronSite.size();
  }

  int qn_label = 1;
  for (size_t i = 0; i < ElectronSite.size(); i++) {
    if (i % sitenumber_perhole == sitenumber_perhole / 2) {
      stat_labs[ElectronSite[i]] = 3;
    } else {
      stat_labs[ElectronSite[i]] = qn_label;
      qn_label = 3 - qn_label;
    }
  }
  if (world.rank() == 0) {
    if (IsPathExist(kMpsPath)) {//mps only can be load from file
      if (N == GetNumofMps()) {
        cout << "The number of mps files is consistent with mps size." << endl;
        cout << "Directly use mps from files." << endl;
      } else {
        cout << "mps file number do not right" << endl;
        cout << "Initial mps as direct product state." << endl;
        qlmps::DirectStateInitMps(mps, stat_labs);
        mps.Dump(sweep_params.mps_path, true);
      }
    } else {
      cout << " no mps file" << endl;
      cout << "Initial mps as direct product state." << endl;
      qlmps::DirectStateInitMps(mps, stat_labs);
      mps.Dump(sweep_params.mps_path, true);
    }
  }
  double e0;
  if (world.size() == 1) {
    e0 = qlmps::TwoSiteFiniteVMPS(mps, mpo, sweep_params);
  } else {
    e0 = qlmps::TwoSiteFiniteVMPS(mps, mpo, sweep_params, world);
  }

  if (world.rank() == 0) {
    std::cout << "E0/site: " << e0 / N << std::endl;
    endTime = clock();
    cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  }
  return 0;

}
