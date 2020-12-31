//
// Created by 付聪 on 2017/6/21.
//

#include <chrono>

#include "index_random.h"
#include "index_ssg.h"
#include "util.h"

void save_result(char* filename, std::vector<std::vector<unsigned> >& results) {
  std::cout << "Save result to " << filename << std::endl;

  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void load_result(const char* filename, unsigned query_num,
                 std::vector<std::vector<unsigned>>& results) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "Open last result " << filename << " failed" << std::endl;
        return;
    }
    results.resize(query_num);
    for (unsigned i = 0; i < query_num; ++i) {
        unsigned GK = 0;
        in.read((char*)&GK, sizeof(unsigned));
        results[i].resize(GK);
        in.read((char*)results[i].data(), GK * sizeof(unsigned));
    }
    in.close();
}

struct GroundTruth {
    unsigned truthItemNum;
    unsigned queryNum;
    unsigned* data;

    void load(const char* filename) {
        data = efanna2e::load_data<unsigned>(filename, queryNum, truthItemNum);
        std::cout << "ground truth query num: " << queryNum << std::endl;
        std::cout << "ground truth item num per query: " << truthItemNum << std::endl;
    }

    void recallRate(const std::vector<std::vector<unsigned>>& res) {
        // const unsigned TOPK = res.front().size() / 2;
        const unsigned TOPK = 100;
        assert(TOPK <= truthItemNum);
        assert(res.size() <= queryNum);
        float avgRecallVal = 0;
        for (unsigned qi = 0; qi < res.size(); ++qi) {
            auto truth = data + qi * truthItemNum;
            unsigned recallNum = 0;
            for (auto docId : res[qi]) {
                for (unsigned j = 0; j < TOPK; ++j) {
                    if (truth[j] == docId) {
                        ++recallNum;
                        break;
                    }
                }
            }
            auto recallRateVal = (float)recallNum / TOPK;
            // recallRate.push_back(recallRateVal);
            avgRecallVal += recallRateVal;
        }
        auto recall = avgRecallVal / res.size();
        std::cout << "recall(top" << TOPK << ") : " << recall << std::endl;
    }
};

int main(int argc, char** argv) {
#ifdef __AVX__
    std::cout << "__AVX__" << std::endl;
#endif

  if (argc < 7) {
    std::cout << "./run data_file query_file ssg_path L K result_path [seed]"
              << std::endl;
    exit(-1);
  }

  std::cerr << "Data Path: " << argv[1] << std::endl;

  unsigned points_num, dim;
  float* data_load = nullptr;
  data_load = efanna2e::load_data<float>(argv[1], points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim);

  std::cerr << "Query Path: " << argv[2] << std::endl;

  unsigned query_num, query_dim;
  float* query_load = nullptr;
  query_load = efanna2e::load_data<float>(argv[2], query_num, query_dim);
  query_load = efanna2e::data_align(query_load, query_num, query_dim);

  assert(dim == query_dim);

    std::vector<std::vector<unsigned>> lastResult;
    if (argc == 8) {
        std::cerr << "Last result path: " << argv[7] << std::endl;
        load_result(argv[7], query_num, lastResult);
    }

  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexSSG index(dim, points_num, efanna2e::FAST_L2,
                           (efanna2e::Index*)(&init_index));

  std::cerr << "SSG Path: " << argv[3] << std::endl;
  //std::cerr << "Result Path: " << argv[6] << std::endl;
  std::cerr << "Ground Truth Path: " << argv[6] << std::endl;
  std::cout << "Query num: " << query_num << std::endl;

  index.Load(argv[3]);
  index.OptimizeGraph(data_load);

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);

  std::cerr << "L = " << L << ", ";
  std::cerr << "K = " << K << std::endl;

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  std::function<void(unsigned)> loadInitIdFunc;
  auto searchFunc = [&]() {
      std::vector<std::vector<unsigned> > res(query_num);
      for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

      // Warm up
      for (int loop = 0; loop < 3; ++loop) {
          for (unsigned i = 0; i < 10; ++i) {
              loadInitIdFunc(i);
              index.SearchWithOptGraph(query_load + i * dim, K, paras,
                                       res[i].data());
          }
      }

      auto s = std::chrono::high_resolution_clock::now();
      unsigned totalHops = 0;
      unsigned totalVisit = 0;
      for (unsigned i = 0; i < query_num; i++) {
          loadInitIdFunc(i);
          index.SearchWithOptGraph(query_load + i * dim, K, paras,
                                   res[i].data());
          totalHops += index.getHops();
          totalVisit += index.getVisitNum();
      }
      auto e = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> diff = e - s;
      std::cerr << "Search Time: " << diff.count() << std::endl;
      std::cerr << "QPS: " << query_num / diff.count() << std::endl;
      std::cerr << "AVG latency(ms): " << 1000 * diff.count() / query_num
                << std::endl;
      std::cerr << "AVG visit num: " << (float) totalVisit / query_num
                << std::endl;
      std::cerr << "AVG hop num: " << (float) totalHops / query_num
                << std::endl;

      GroundTruth truth;
      truth.load(argv[6]);
      truth.recallRate(res);
      if (argc == 8 && lastResult.empty()) {
          save_result(argv[7], res);
      }
  };
  if (lastResult.empty()) {
      // index.loadInitIds(L);
      loadInitIdFunc = [&index, L](unsigned i) { index.loadInitIds(L); };
      searchFunc();
  } else {
      auto& initIds = index.initIds();
      loadInitIdFunc = [&initIds, &lastResult](unsigned qi) {
          initIds = lastResult[qi];
          initIds.resize(initIds.size()/2);
          // initIds.resize(1);
      };
      searchFunc();
  }
  return 0;
}
