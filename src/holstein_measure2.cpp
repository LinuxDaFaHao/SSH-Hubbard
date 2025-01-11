/*
    holstein_measure2.cpp
    for measure2 two point function for the holstein model.
    usage:
      mpirun -n 4 ./holstein_measure2 params.json
    note: processor number must be 4.
        Optional arguments:
      --start=
      --end=
*/
#include "gqdouble.h"
#include "operators.h"
#include "params_case.h"

#include "qlmps/qlmps.h"
#include "qlten/qlten.h"
#include <time.h>
#include <stdlib.h>

#include "myutil.h"
#include "my_measure_appendix.h"

#include "qlten/utility/timer.h"



using std::cout;
using std::endl;
using std::vector;
using FiniteMPST = qlmps::FiniteMPS<TenElemT, U1U1QN>;
using qlmps::SiteVec;
using qlmps::MeasureOneSiteOp;
using qlten::Timer;

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  clock_t startTime, endTime;
  startTime = clock();

  size_t beginx;
  size_t endx;
  bool start_argument_has = Parser(argc, argv, beginx, endx);

  CaseParams params(argv[1]);
  cout << "This program will measure the two point function of Holstein-Hubbard model." << std::endl;
  size_t Lx = params.Lx, Ly = params.Ly, Np = params.Np;
  size_t N = (1 + Np) * (Lx * Ly);
  if (GetNumofMps() != N) {
    std::cout << "The number of mps files are inconsistent with mps size!" << std::endl;
    exit(1);
  }

  if (!start_argument_has) {
    beginx = Lx / 4;
    endx = beginx + Lx / 2;
  }

  OperatorInitial();

  std::vector<IndexT> pb_out_set(N);
  std::vector<size_t> Fsite_set(Lx * Ly);
  std::vector<size_t> Bsite_set(N - Lx * Ly);
  auto iterF = Fsite_set.begin();
  auto iterB = Bsite_set.begin();

  for (size_t i = 0; i < N; ++i) {
    if (i % (Np + 1) == 0) {
      pb_out_set[i] = pb_outF;
      *iterF = i;
      iterF++;
    } else {
      pb_out_set[i] = pb_outB;
      *iterB = i;
      iterB++;
    }
  }
  cout << "The Fermion sites: " << endl;
  Show(Fsite_set);
  cout << '\n';
  cout << "The Boson sites:" << endl;
  Show(Bsite_set);

  SiteVec<TenElemT, U1U1QN> sites = SiteVec<TenElemT, U1U1QN>(pb_out_set);
  FiniteMPST mps(sites);


  qlten::hp_numeric::SetTensorManipulationThreads(params.TotalThreads);

  vector<vector<size_t>> two_point_sites_setF;

  two_point_sites_setF.reserve(Ly * (endx - beginx));
  for (size_t y = 0; y < Ly; ++y) {
    size_t site1F = beginx * Ly + y;
    for (size_t x = beginx + 1; x < endx; ++x) {
      size_t site2F = x * Ly + y;
      two_point_sites_setF.push_back({Fsite_set[site1F], Fsite_set[site2F]});
    }
  }

  std::string file_name_postfix;
  if (start_argument_has) {
    file_name_postfix = "begin" + std::to_string(beginx) + "end" + std::to_string(endx);
  } else {
    file_name_postfix = "";
  }

  Timer twosite_timer("measure two site operators");
  MeasureTwoSiteOp(mps, sz, sz, two_point_sites_setF, Ly, "szsz" + file_name_postfix, comm);
  MeasureTwoSiteOp(mps, sp, sm, two_point_sites_setF, Ly, "spsm" + file_name_postfix, comm);
  MeasureTwoSiteOp(mps, sm, sp, two_point_sites_setF, Ly, "smsp" + file_name_postfix, comm);
  MeasureTwoSiteOp(mps, nf, nf, two_point_sites_setF, Ly, "nfnf" + file_name_postfix, comm);
  MeasureTwoSiteOp(mps, cupccdnc, cdnacupa, two_point_sites_setF, Ly, "onsitesc" + file_name_postfix, comm);
  qlmps::MeasureTwoSiteFermionOp(mps,
                                  bupc,
                                  bupa,
                                  two_point_sites_setF,
                                  Ly,
                                  "single_particle" + file_name_postfix,
                                  comm); // correlation <c^dag_spinup(i) c_spinup(j)>
  qlmps::MeasureTwoSiteFermionOp(mps,
                                  -Fbdnc,
                                  Fbdna,
                                  two_point_sites_setF,
                                  Ly,
                                  "single_particle" + file_name_postfix,
                                  comm);// correlation <c^dag_spindown(i) c_spindown(j)>
  cout << "measured two point function.<====" << endl;
  twosite_timer.PrintElapsed();

  endTime = clock();
  cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

  return 0;

}

