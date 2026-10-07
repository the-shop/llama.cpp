#include "llama-expert-pool.h"

#include "llama-impl.h"
#include "llama-model-loader.h"

#include <algorithm>

#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/param.h>
#include <unistd.h>

std::shared_ptr<llama_expert_pool> llama_expert_pool::from_env(int n_layer, int n_expert) {
    const char * gib = getenv("LLAMA_EXPERT_POOL_GIB");
    if (gib == nullptr || atof(gib) <= 0.0) {
        return nullptr;
    }
    const char * thr = getenv("LLAMA_EXPERT_POOL_THREADS");
    return std::make_shared<llama_expert_pool>(atof(gib), thr ? std::max(1, atoi(thr)) : 8, n_layer, n_expert);
}

llama_expert_pool::llama_expert_pool(double gib, int n_threads, int n_layer, int n_expert)
    : gib(gib), n_threads(n_threads), n_layer(n_layer), n_expert(n_expert), layers(n_layer) {
    const char * every = getenv("LLAMA_EXPERT_POOL_LOG_EVERY");
    log_every = every ? strtoull(every, nullptr, 10) : 4800;
}

llama_expert_pool::~llama_expert_pool() {
    if (n_calls_lru + n_calls_window > 0) {
        log_stats("final");
    }
    if (trace) {
        fclose(trace);
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    cv_work.notify_all();
    for (auto & t : threads) {
        t.join();
    }
    for (int p = 0; p < N_PROJ; ++p) {
        if (buf[p]) {
            ggml_backend_buffer_free(buf[p]);
        }
        if (mem[p]) {
            munmap(mem[p], mem_size[p]);
        }
    }
    if (ctx) {
        ggml_free(ctx);
    }
    for (int fd : fds) {
        close(fd);
    }
}

void llama_expert_pool::add_source(const llama_model_loader & ml, int il, proj p, const std::string & name) {
    const auto * w = ml.get_weight(name.c_str());
    if (w == nullptr) {
        throw std::runtime_error("expert pool: missing tensor " + name);
    }

    // our own fd per file: F_NOCACHE is a property of the open file, the loader's fd must keep caching
    char path[MAXPATHLEN];
    if (fcntl(ml.files[w->idx]->file_id(), F_GETPATH, path) == -1) {
        throw std::runtime_error("expert pool: F_GETPATH failed: " + std::string(strerror(errno)));
    }
    int fd = -1;
    for (size_t i = 0; i < paths.size(); ++i) {
        if (paths[i] == path) {
            fd = fds[i];
        }
    }
    if (fd < 0) {
        fd = open(path, O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error(std::string("expert pool: cannot open ") + path);
        }
        fcntl(fd, F_NOCACHE, 1);
        fcntl(fd, F_RDAHEAD, 0);
        fds.push_back(fd);
        paths.push_back(path);
    }

    const ggml_tensor * t = w->tensor;
    if (t->ne[2] != n_expert || t->ne[3] != 1) {
        throw std::runtime_error("expert pool: unexpected shape of " + name);
    }
    source & s = layers[il].src[p];
    s.fd     = fd;
    s.offset = w->offs;
    s.stride = t->nb[2];
    s.type   = t->type;
    for (int i = 0; i < 3; ++i) {
        s.ne[i] = t->ne[i];
    }
    if (il > 0 && layers[0].src[p].fd >= 0) {
        const source & s0 = layers[0].src[p];
        if (s0.type != s.type || s0.stride != s.stride || s0.ne[0] != s.ne[0] || s0.ne[1] != s.ne[1]) {
            throw std::runtime_error("expert pool: expert tensors differ between layers");
        }
    }
}

void llama_expert_pool::init(bool dry) {
    for (int il = 0; il < n_layer; ++il) {
        for (int p = 0; p < N_PROJ; ++p) {
            if (layers[il].src[p].fd < 0) {
                throw std::runtime_error("expert pool: layer " + std::to_string(il) + " has no expert sources");
            }
        }
    }

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU || ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            dev = d;
            break;
        }
    }
    if (dev == nullptr) {
        throw std::runtime_error("expert pool: no GPU device");
    }

    size_t slot_bytes = 0;
    for (int p = 0; p < N_PROJ; ++p) {
        slot_bytes += layers[0].src[p].stride;
    }
    n_slots = (int64_t) (gib * 1024.0 * 1024.0 * 1024.0 / slot_bytes);

    // the window holds one layer, the LRU at least a decode call's experts (fewer than 32 rows x top-k)
    const int64_t n_min = 2*(int64_t) n_expert;
    if (n_slots < n_min) {
        throw std::runtime_error(format("expert pool: %.1f GiB gives %" PRId64 " slots, need at least %" PRId64 " (%.1f GiB)",
                    gib, n_slots, n_min, (double) n_min*slot_bytes/(1024.0*1024.0*1024.0)));
    }

    ggml_init_params params = { ggml_tensor_overhead()*N_PROJ, nullptr, true };
    ctx = ggml_init(params);

    const size_t page = (size_t) getpagesize();
    static const char * names[N_PROJ] = { "expert_pool.gate", "expert_pool.up", "expert_pool.down" };
    for (int p = 0; p < N_PROJ; ++p) {
        const source & s = layers[0].src[p];
        pool[p] = ggml_new_tensor_3d(ctx, s.type, s.ne[0], s.ne[1], n_slots);
        ggml_set_name(pool[p], names[p]);
        GGML_ASSERT(pool[p]->nb[2] == s.stride);

        if (dry) {
            // a no_alloc model (--fit): a dummy buffer so the scheduler places the matmuls without allocating them
            buf[p] = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), 0);
            ggml_backend_buffer_set_usage(buf[p], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            pool[p]->buffer = buf[p];
            continue;
        }

        mem_size[p] = (ggml_nbytes(pool[p]) + page - 1) / page * page;
        mem[p] = mmap(nullptr, mem_size[p], PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (mem[p] == MAP_FAILED) {
            mem[p] = nullptr;
            throw std::runtime_error("expert pool: mmap failed");
        }
        buf[p] = ggml_backend_dev_buffer_from_host_ptr(dev, mem[p], mem_size[p], mem_size[p]);
        if (buf[p] == nullptr) {
            throw std::runtime_error("expert pool: the GPU cannot map the pool memory");
        }
        ggml_backend_buffer_set_usage(buf[p], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (ggml_backend_tensor_alloc(buf[p], pool[p], mem[p]) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("expert pool: tensor alloc failed");
        }
    }

    calls.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        calls[il] = { this, il };
    }
    initialized = true;
    if (dry) {
        return;
    }


    slot_of.assign((size_t) n_layer*n_expert, -1);
    stamp.assign((size_t) n_layer*n_expert, 0);
    key_of.assign(n_slots, -1);
    prev.assign(n_slots, -1);
    next.assign(n_slots, -1);
    window_layer.assign(n_expert, -1);
    n_free_next = n_expert;

    for (int i = 1; i < n_threads; ++i) {
        threads.emplace_back([this] { worker(); });
    }

    if (const char * path = getenv("LLAMA_EXPERT_TRACE")) {
        trace = fopen(path, "wb");
        if (!trace) {
            throw std::runtime_error(std::string("expert pool: cannot open trace file ") + path);
        }
        setvbuf(trace, nullptr, _IOFBF, 1 << 20);
        trace_tok.assign(n_layer, 0);
        const uint32_t hdr[4] = { 1, (uint32_t) n_layer, (uint32_t) n_expert, 0 };
        fwrite("LXTR", 1, 4, trace);
        fwrite(hdr, sizeof(uint32_t), 3, trace); // top_k is written with the first call
        LLAMA_LOG_WARN("%s: tracing the routing to %s\n", __func__, path);
    }

    LLAMA_LOG_WARN("%s: %.2f GiB on %s: %" PRId64 " slots of %.2f MiB (window %d + LRU %" PRId64 "), %d reader threads, %zu files with F_NOCACHE\n",
            __func__, (double) n_slots*slot_bytes/(1024.0*1024.0*1024.0), ggml_backend_dev_name(dev), n_slots, slot_bytes/(1024.0*1024.0),
            n_expert, n_slots - n_expert, n_threads, fds.size());
}

static void llama_expert_pool_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * c = (llama_expert_pool::call *) userdata;
    c->pool->run(dst->src[0], dst, c->il);
}

ggml_tensor * llama_expert_pool::build_ids(ggml_context * ctx0, ggml_tensor * selected_experts, int il) {
    GGML_ASSERT(selected_experts->type == GGML_TYPE_I32);
    ggml_tensor * args[1] = { selected_experts };
    ggml_tensor * ids = ggml_custom_4d(ctx0, GGML_TYPE_I32, selected_experts->ne[0], selected_experts->ne[1], 1, 1,
            args, 1, llama_expert_pool_op, 1, &calls[il]);
    return ids;
}

ggml_tensor * llama_expert_pool::view(ggml_context * ctx0, proj p, int64_t n_tokens) const {
    ggml_tensor * t = pool[p];
    if (window_mode(n_tokens)) {
        return ggml_view_3d(ctx0, t, t->ne[0], t->ne[1], n_expert, t->nb[1], t->nb[2], 0);
    }
    return t;
}

void llama_expert_pool::run(const ggml_tensor * ids, ggml_tensor * dst, int il) {
    const int64_t t0 = ggml_time_us();
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];

    ++cur_stamp;
    std::vector<int32_t> experts;
    for (int64_t t = 0; t < n_tokens; ++t) {
        const int32_t * row = (const int32_t *) ((const char *) ids->data + t*ids->nb[1]);
        for (int64_t j = 0; j < n_used; ++j) {
            const int32_t e = row[j];
            GGML_ASSERT(e >= 0 && e < n_expert);
            const size_t key = (size_t) il*n_expert + e;
            if (stamp[key] != cur_stamp) {
                stamp[key] = cur_stamp;
                experts.push_back(e);
            }
        }
    }

    if (trace) {
        trace_write(ids, il, window_mode(n_tokens));
    }

    std::vector<int32_t> slot_of_expert;
    if (window_mode(n_tokens)) {
        load_window(il, experts);
    } else {
        load_lru(il, experts, slot_of_expert);
    }

    int32_t * out = (int32_t *) dst->data;
    for (int64_t t = 0; t < n_tokens; ++t) {
        const int32_t * row = (const int32_t *) ((const char *) ids->data + t*ids->nb[1]);
        for (int64_t j = 0; j < n_used; ++j) {
            out[t*n_used + j] = window_mode(n_tokens) ? row[j] : slot_of_expert[row[j]];
        }
    }

    t_total += (ggml_time_us() - t0) / 1e6;
    if (window_mode(n_tokens)) {
        ++n_calls_window;
    } else if (++n_calls_lru % (log_every ? log_every : UINT64_MAX) == 0) {
        log_stats("progress");
    }
}

void llama_expert_pool::load_lru(int il, const std::vector<int32_t> & experts, std::vector<int32_t> & slot_of_expert) {
    slot_of_expert.assign(n_expert, -1);
    // hits first: once they are the most recent, eviction from the head cannot take an expert of this call
    for (int32_t e : experts) {
        const int32_t slot = slot_of[(int64_t) il*n_expert + e];
        if (slot >= 0) {
            ++n_hit;
            lru_unlink(slot);
            lru_push_back(slot);
            slot_of_expert[e] = slot;
        }
    }
    for (int32_t e : experts) {
        if (slot_of_expert[e] >= 0) {
            continue;
        }
        ++n_miss;
        const int64_t key = (int64_t) il*n_expert + e;
        int32_t slot;
        if (n_free_next < n_slots) {
            slot = (int32_t) n_free_next++;
        } else {
            slot = head;
            const int64_t old = key_of[slot];
            GGML_ASSERT(stamp[old] != cur_stamp && "expert pool: one call needs more experts than the LRU holds");
            lru_unlink(slot);
            slot_of[old] = -1;
        }
        key_of[slot] = key;
        slot_of[key] = slot;
        queue_load(il, e, slot);
        lru_push_back(slot);
        slot_of_expert[e] = slot;
    }
    run_tasks();
}

void llama_expert_pool::load_window(int il, const std::vector<int32_t> & experts) {
    for (int32_t e : experts) {
        if (window_layer[e] == il) {
            ++n_win_hit;
            continue;
        }
        const int32_t slot = slot_of[(int64_t) il*n_expert + e];
        if (slot >= 0) {
            ++n_win_copy;
            queue_copy(slot, e);
        } else {
            ++n_win_read;
            queue_load(il, e, e);
        }
        window_layer[e] = il;
    }
    run_tasks();
}

void llama_expert_pool::queue_load(int il, int expert, int slot) {
    for (int p = 0; p < N_PROJ; ++p) {
        const source & s = layers[il].src[p];
        tasks.push_back({ (uint8_t *) mem[p] + (size_t) slot*s.stride, nullptr, s.fd, s.offset + (size_t) expert*s.stride, s.stride });
    }
}

void llama_expert_pool::queue_copy(int from_slot, int to_slot) {
    for (int p = 0; p < N_PROJ; ++p) {
        const size_t n = layers[0].src[p].stride;
        tasks.push_back({ (uint8_t *) mem[p] + (size_t) to_slot*n, (const uint8_t *) mem[p] + (size_t) from_slot*n, -1, 0, n });
    }
}

static void llama_expert_pool_do(const llama_expert_pool::task & t) {
    if (t.src) {
        memcpy(t.dst, t.src, t.n);
        return;
    }
    size_t done = 0;
    while (done < t.n) {
        const ssize_t got = pread(t.fd, t.dst + done, t.n - done, (off_t) (t.offset + done));
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            GGML_ABORT("expert pool: pread failed: %s", got < 0 ? strerror(errno) : "end of file");
        }
        done += (size_t) got;
    }
}

void llama_expert_pool::run_tasks() {
    if (tasks.empty()) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    for (const auto & t : tasks) {
        if (!t.src) {
            bytes_read += t.n;
        }
    }
    std::unique_lock<std::mutex> lock(mtx);
    next_task = 0;
    n_done = 0;
    ++generation;
    cv_work.notify_all();
    while (next_task < tasks.size()) {
        const task t = tasks[next_task++];
        lock.unlock();
        llama_expert_pool_do(t);
        lock.lock();
        ++n_done;
    }
    cv_done.wait(lock, [this] { return n_done == tasks.size(); });
    tasks.clear();
    t_read += (ggml_time_us() - t0) / 1e6;
}

void llama_expert_pool::worker() {
    uint64_t seen = 0;
    std::unique_lock<std::mutex> lock(mtx);
    while (true) {
        cv_work.wait(lock, [&] { return stop || generation != seen; });
        if (stop) {
            return;
        }
        seen = generation;
        while (next_task < tasks.size()) {
            const task t = tasks[next_task++];
            lock.unlock();
            llama_expert_pool_do(t);
            lock.lock();
            if (++n_done == tasks.size()) {
                cv_done.notify_one();
            }
        }
    }
}

void llama_expert_pool::lru_unlink(int slot) {
    const int32_t p = prev[slot], n = next[slot];
    if (p >= 0) next[p] = n; else if (head == slot) head = n;
    if (n >= 0) prev[n] = p; else if (tail == slot) tail = p;
    prev[slot] = next[slot] = -1;
}

void llama_expert_pool::lru_push_back(int slot) {
    prev[slot] = tail;
    next[slot] = -1;
    if (tail >= 0) next[tail] = slot; else head = slot;
    tail = slot;
}

void llama_expert_pool::log_stats(const char * why) {
    if (trace) {
        fflush(trace);
    }
    const uint64_t n = n_hit + n_miss;
    LLAMA_LOG_WARN("expert_pool %s: decode calls %" PRIu64 " experts %" PRIu64 " hit %.1f%% | prefill calls %" PRIu64
            " window hit %" PRIu64 " copy %" PRIu64 " read %" PRIu64 " | read %.2f GiB in %.2f s (%.2f GB/s) | op %.2f s\n",
            why, n_calls_lru, n, n ? 100.0*n_hit/n : 0.0, n_calls_window, n_win_hit, n_win_copy, n_win_read,
            bytes_read/(1024.0*1024.0*1024.0), t_read, t_read > 0 ? bytes_read/1e9/t_read : 0.0, t_total);
}

void llama_expert_pool::trace_write(const ggml_tensor * ids, int il, bool prefill) {
    const int64_t k = ids->ne[0];
    if (ftell(trace) == 16) {
        const uint32_t top_k = (uint32_t) k;
        fwrite(&top_k, sizeof(top_k), 1, trace);
    }
    std::vector<uint8_t> rec(8 + 2*k);
    for (int64_t t = 0; t < ids->ne[1]; ++t) {
        const uint32_t tok = trace_tok[il]++;
        const uint16_t layer = (uint16_t) il;
        memcpy(rec.data(), &tok, 4);
        memcpy(rec.data() + 4, &layer, 2);
        rec[6] = (uint8_t) k;
        rec[7] = prefill ? 1 : 0;
        const int32_t * row = (const int32_t *) ((const char *) ids->data + t*ids->nb[1]);
        for (int64_t j = 0; j < k; ++j) {
            const uint16_t e = (uint16_t) row[j];
            memcpy(rec.data() + 8 + 2*j, &e, 2);
        }
        fwrite(rec.data(), 1, rec.size(), trace);
    }
}

llama_io_pool::llama_io_pool(int n_threads) {
    for (int i = 0; i < n_threads; ++i) {
        threads.emplace_back([this] { worker(); });
    }
}

llama_io_pool::~llama_io_pool() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        stop = true;
    }
    cv_work.notify_all();
    for (auto & t : threads) {
        t.join();
    }
}

void llama_io_pool::submit(std::vector<job> & jobs, group & g, bool front) {
    if (jobs.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        g.remaining += (int) jobs.size();
        std::vector<job *> ptrs;
        for (auto & j : jobs) {
            j.g = &g;
            j.state = 0;
            ptrs.push_back(&j);
        }
        if (front) {
            q.insert(q.begin(), ptrs.begin(), ptrs.end());
        } else {
            q.insert(q.end(), ptrs.begin(), ptrs.end());
        }
    }
    cv_work.notify_all();
}

void llama_io_pool::wait(group & g) {
    std::unique_lock<std::mutex> lock(mtx);
    cv_done.wait(lock, [&] { return g.remaining == 0; });
}

bool llama_io_pool::cancel(job & j) {
    std::lock_guard<std::mutex> lock(mtx);
    if (j.state != 0) {
        return false;
    }
    q.erase(std::find(q.begin(), q.end(), &j));
    j.state = 3;
    if (--j.g->remaining == 0) {
        cv_done.notify_all();
    }
    return true;
}

void llama_io_pool::run(std::vector<job> & jobs) {
    group g;
    submit(jobs, g, true);
    wait(g);
}

static void llama_io_do(const llama_io_pool::job & t) {
    if (t.src) {
        memcpy(t.dst, t.src, t.n);
        return;
    }
    size_t done = 0;
    while (done < t.n) {
        const ssize_t got = pread(t.fd, t.dst + done, t.n - done, (off_t) (t.offset + done));
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            GGML_ABORT("expert pool: pread failed: %s", got < 0 ? strerror(errno) : "end of file");
        }
        done += (size_t) got;
    }
}

void llama_io_pool::worker() {
    std::unique_lock<std::mutex> lock(mtx);
    while (true) {
        cv_work.wait(lock, [&] { return stop || !q.empty(); });
        if (stop) {
            return;
        }
        job * j = q.front();
        q.pop_front();
        j->state = 1;
        lock.unlock();
        llama_io_do(*j);
        if (!j->src) {
            bytes_read += j->n;
        }
        lock.lock();
        j->state = 2;
        if (--j->g->remaining == 0) {
            cv_done.notify_all();
        }
    }
}

int llama_open_nocache(const llama_model_loader & ml, const std::string & name, size_t & offset, std::string & path) {
    const auto * w = ml.get_weight(name.c_str());
    if (w == nullptr) {
        throw std::runtime_error("missing tensor " + name);
    }
    char buf[MAXPATHLEN];
    if (fcntl(ml.files[w->idx]->file_id(), F_GETPATH, buf) == -1) {
        throw std::runtime_error("F_GETPATH failed: " + std::string(strerror(errno)));
    }
    path = buf;
    offset = w->offs;
    const int fd = open(buf, O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("cannot open " + path);
    }
    // F_NOCACHE is a property of the open file, so the loader's own fd keeps caching
    fcntl(fd, F_NOCACHE, 1);
    fcntl(fd, F_RDAHEAD, 0);
    return fd;
}

std::shared_ptr<llama_row_store> llama_row_store::from_env(const llama_model_loader & ml, const std::string & name) {
    const char * on = getenv("LLAMA_PLE_PREAD");
    if (on == nullptr || atoi(on) == 0) {
        return nullptr;
    }
    const auto * w = ml.get_weight(name.c_str());
    if (w == nullptr) {
        return nullptr;
    }
    const char * mib = getenv("LLAMA_PLE_CACHE_MIB");
    const char * thr = getenv("LLAMA_PLE_THREADS");
    size_t offset;
    std::string path;
    const int fd = llama_open_nocache(ml, name, offset, path);
    const ggml_tensor * t = w->tensor;
    GGML_ASSERT(ggml_n_dims(t) == 2);
    return std::make_shared<llama_row_store>(fd, offset, t->type, t->ne[0], t->ne[1],
            (size_t) (mib ? atof(mib) : 1024.0) * 1024 * 1024, thr ? std::max(1, atoi(thr)) : 8);
}

llama_row_store::llama_row_store(int fd, size_t offset, ggml_type type, int64_t ne0, int64_t n_rows, size_t cache_bytes, int n_threads)
    : ne0(ne0), fd(fd), offset(offset), type(type), n_rows(n_rows), row_bytes(ggml_row_size(type, ne0)), io(n_threads) {
    GGML_ASSERT(ggml_get_type_traits(type)->to_float != nullptr || type == GGML_TYPE_F32);
    capacity = std::max<int64_t>(1, (int64_t) (cache_bytes / row_bytes));
    data.resize((size_t) capacity * row_bytes);
    row_of.assign(capacity, -1);
    ref.assign(capacity, 0);
    stamp.assign(capacity, 0);
    LLAMA_LOG_WARN("%s: %" PRId64 " rows of %zu bytes (%.0f MiB), %d reader threads, F_NOCACHE\n",
            __func__, capacity, row_bytes, capacity*row_bytes/(1024.0*1024.0), n_threads);
}

llama_row_store::~llama_row_store() {
    const uint64_t n = n_hit + n_miss;
    if (n > 0) {
        LLAMA_LOG_WARN("ple_rows final: rows %" PRIu64 " hit %.1f%% | read %.2f MiB in %.2f s\n",
                n, 100.0*n_hit/n, io.bytes_read/(1024.0*1024.0), t_read);
    }
    close(fd);
}

void llama_row_store::gather(const int32_t * rows, size_t n, float * out) {
    ++cur;
    std::vector<int32_t> slots(n);
    std::vector<llama_io_pool::job> jobs;
    for (size_t i = 0; i < n; ++i) {
        const int32_t r = rows[i];
        GGML_ASSERT(r >= 0 && r < n_rows);
        auto it = slot_of.find(r);
        if (it != slot_of.end()) {
            const int32_t s = it->second;
            if (stamp[s] != cur) {
                ++n_hit;
            }
            ref[s] = 1;
            stamp[s] = cur;
            slots[i] = s;
            continue;
        }
        ++n_miss;
        // CLOCK: a used bit buys one more sweep; rows of this call are never taken
        int64_t s;
        for (int64_t steps = 0;; ++steps) {
            GGML_ASSERT(steps <= 2*capacity && "PLE row cache is smaller than one batch's rows");
            s = hand;
            hand = (hand + 1) % capacity;
            if (row_of[s] < 0) {
                break;
            }
            if (stamp[s] == cur) {
                continue;
            }
            if (ref[s]) {
                ref[s] = 0;
                continue;
            }
            slot_of.erase(row_of[s]);
            break;
        }
        row_of[s] = r;
        slot_of[r] = (int32_t) s;
        ref[s] = 1;
        stamp[s] = cur;
        slots[i] = (int32_t) s;
        jobs.push_back({ data.data() + (size_t) s*row_bytes, nullptr, fd, offset + (size_t) r*row_bytes, row_bytes });
    }
    if (!jobs.empty()) {
        const int64_t t0 = ggml_time_us();
        io.run(jobs);
        t_read += (ggml_time_us() - t0) / 1e6;
    }
    const auto to_float = ggml_get_type_traits(type)->to_float;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t * src = data.data() + (size_t) slots[i]*row_bytes;
        if (type == GGML_TYPE_F32) {
            memcpy(out + i*ne0, src, row_bytes);
        } else {
            to_float(src, out + i*ne0, ne0);
        }
    }
}
