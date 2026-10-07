#pragma once

// Experimental SSD expert streaming (LLAMA_EXPERT_POOL_GIB=N): routed expert weights stay in the GGUF files and are
// read with F_NOCACHE pread into a fixed GPU slot pool. Between the router top-k and MUL_MAT_ID a CPU custom op
// loads the routed experts and turns expert ids into slot ids; MUL_MAT_ID then runs on the GPU over the pool.
//   decode (n_tokens < 32): LRU over (layer, expert) in slots [n_expert, n_slots), ids remapped to slots
//   prefill (n_tokens >= 32): slots [0, n_expert) hold one layer's experts, expert e in slot e (the Metal mm_id
//   kernel takes at most 1024 experts), ids unchanged

#include "ggml.h"
#include "ggml-backend.h"

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct llama_model_loader;

struct llama_expert_pool {
    enum proj { GATE = 0, UP = 1, DOWN = 2, N_PROJ = 3 };

    // the pool is enabled by the env, nullptr otherwise
    static std::shared_ptr<llama_expert_pool> from_env(int n_layer, int n_expert);

    llama_expert_pool(double gib, int n_threads, int n_layer, int n_expert);
    ~llama_expert_pool();

    // remember where layer il's expert tensor of `p` lives in the model files (the tensor itself is not loaded)
    void add_source(const llama_model_loader & ml, int il, proj p, const std::string & name);

    // allocate the pool on the first GPU device, after every layer has its sources (dry: tensors only, no memory)
    void init(bool dry);

    bool has_layer(int il) const { return initialized && il >= 0 && il < n_layer; }

    // slot ids for MUL_MAT_ID over view(); runs the loads on the CPU when the graph reaches it
    ggml_tensor * build_ids(ggml_context * ctx, ggml_tensor * selected_experts, int il);

    // the pool tensor of `p` as the mul_mat_id weights for a call with n_tokens rows
    ggml_tensor * view(ggml_context * ctx, proj p, int64_t n_tokens) const;

    static bool window_mode(int64_t n_tokens) { return n_tokens >= 32; }

    struct source {
        int     fd     = -1;
        size_t  offset = 0; // expert 0 in the file
        size_t  stride = 0; // bytes of one expert
        int64_t ne[3]  = {0, 0, 0};
        ggml_type type = GGML_TYPE_COUNT;
    };

    struct layer_info {
        source src[N_PROJ];
    };

    struct call {
        llama_expert_pool * pool;
        int il;
    };

    struct task {
        uint8_t     * dst;
        const uint8_t * src; // memcpy from another slot when set
        int           fd;
        size_t        offset;
        size_t        n;
    };

    void run(const ggml_tensor * ids, ggml_tensor * dst, int il);

private:
    void load_lru (int il, const std::vector<int32_t> & experts, std::vector<int32_t> & slot_of_expert);
    void load_window(int il, const std::vector<int32_t> & experts);
    void queue_load(int il, int expert, int slot);
    void queue_copy(int from_slot, int to_slot);
    void run_tasks();
    void worker();
    void lru_unlink(int slot);
    void lru_push_back(int slot);
    void log_stats(const char * why);

    double gib;
    int n_threads;
    int n_layer;
    int n_expert;
    int64_t n_slots = 0;
    bool initialized = false;

    std::vector<layer_info> layers;
    std::vector<int> fds; // owned, one per model file
    std::vector<std::string> paths;

    ggml_context * ctx = nullptr;
    ggml_tensor * pool[N_PROJ] = {nullptr, nullptr, nullptr};
    ggml_backend_buffer_t buf[N_PROJ] = {nullptr, nullptr, nullptr};
    void * mem[N_PROJ] = {nullptr, nullptr, nullptr};
    size_t mem_size[N_PROJ] = {0, 0, 0};

    std::vector<call> calls;

    // LRU over slots [n_expert, n_slots), keyed by il*n_expert + e
    std::vector<int32_t> slot_of; // key -> slot or -1
    std::vector<int64_t> key_of;  // slot -> key or -1
    std::vector<int32_t> prev, next;
    int32_t head = -1, tail = -1; // head = least recently used
    int64_t n_free_next = 0;      // slots [n_free_next, n_slots) never used yet
    std::vector<int32_t> window_layer; // window slot e holds expert e of this layer, -1 if none
    std::vector<uint32_t> stamp;       // per key: last call that touched it (dedup within a call)
    uint32_t cur_stamp = 0;

    // reader threads
    std::vector<std::thread> threads;
    std::mutex mtx;
    std::condition_variable cv_work, cv_done;
    std::vector<task> tasks;
    size_t next_task = 0, n_done = 0;
    uint64_t generation = 0;
    bool stop = false;

    // stats
    uint64_t n_calls_lru = 0, n_calls_window = 0, n_hit = 0, n_miss = 0, n_win_hit = 0, n_win_copy = 0, n_win_read = 0;
    uint64_t bytes_read = 0;
    double t_read = 0.0, t_total = 0.0;
    uint64_t log_every = 0;
};
