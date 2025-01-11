#include "gqdouble.h"
#include "operators.h"
#include <time.h>
#include <vector>
#include <stdlib.h>     // system
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
  mpi::environment env(mpi::threading::multiple);
  if (env.thread_level() < mpi::threading::multiple) {
    std::cout << "thread level of env is not right." << std::endl;
    env.abort(-1);
  }
  mpi::communicator world;
  CaseParams params(argv[1]);
  unsigned Lx = params.Lx, Ly = params.Ly, Np = params.Np;
  unsigned N = Lx * Ly + (2 * Lx * Ly - Ly - Lx) * Np;
  cout << "System size = (" << Lx << "," << Ly << ")" << endl;
  cout << "The number of electron sites =" << Lx * Ly << endl;
  cout << "The number of phonon pseudosite (per bond) =" << Np << endl;
  cout << "The number of phonon pseudosite (total) =" << (2 * Lx * Ly - Ly - Lx) * Np << endl;
  cout << "The total number of sites = " << N << endl;
  float t = params.t, g = params.g, U = params.U, omega = params.omega;
  cout << "Model parameter: t =" << t << ", g =" << g << ", U =" << U << ",omega=" << omega << endl;
  clock_t startTime, endTime;
  startTime = clock();
  OperatorInitial();
  vector<IndexT> pb_out_set(N);
  vector<long> Tx(N, -1), Ty(N, -1), ElectronSite(Lx * Ly);
  auto iter = ElectronSite.begin();
  // translation along x(for electron) and translation along y(for electron);
  const size_t total_Ly = (2 * Np + 1) * Ly - Np;
  for (size_t i = 0; i < N; ++i) {
    size_t residue = i % total_Ly;
    if (residue < (Np + 1) * Ly && residue % (Np + 1) == 0) {
      pb_out_set[i] = pb_outF;
      *iter = i;
      iter++;
    } else pb_out_set[i] = pb_outB;
  }
  SiteVec<TenElemT, U1U1QN> sites = SiteVec<TenElemT, U1U1QN>(pb_out_set);
  MPO<Tensor> mpo(N);
  const std::string kMpoPath = "mpo";
  const std::string kMpoTenBaseName = "mpo_ten";
  for (size_t i = 0; i < mpo.size(); i++) {
    std::string filename = kMpoPath + "/" +
        kMpoTenBaseName + std::to_string(i) + "." + kQLTenFileSuffix;
    mpo.LoadTen(i, filename);
  }

  cout << "MPO loaded." << endl;
  using FiniteMPST = qlmps::FiniteMPS<TenElemT, U1U1QN>;
  FiniteMPST mps(sites);

  if (rank == 0) {
    if (params.TotalThreads > 2) {

      qlten::hp_numeric::SetTensorManipulationThreads(params.TotalThreads - 2);
    } else {

      qlten::hp_numeric::SetTensorManipulationThreads(params.TotalThreads);
    }
  } else {

    qlten::hp_numeric::SetTensorManipulationThreads(params.TotalThreads);
  }
  qlmps::FiniteVMPSSweepParams sweep_params(
      params.Sweeps,
      params.Dmin, params.Dmax, params.CutOff,
      qlmps::LanczosParams(params.LanczErr, params.MaxLanczIter),
      params.noise
  );
  if (IsPathExist(kMpsPath)) {//mps only can be load from file
    if (N == GetNumofMps()) {
      cout << "The number of mps files is consistent with mps size." << endl;
      cout << "Directly use mps from files." << endl;
    } else {
      cout << "mps file number do not right" << endl;
      env.abort(-1);
    }
  } else {
    cout << " no mps file" << endl;
    env.abort(-1);
  }
  auto e0 = qlmps::TwoSiteFiniteVMPS2(mps, mpo, sweep_params, comm);
  if (rank == 0) {
    std::cout << "E0/site: " << e0 / N << std::endl;
    endTime = clock();
    cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  }
  return 0;

}
