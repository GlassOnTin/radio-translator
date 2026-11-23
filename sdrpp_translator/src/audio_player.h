// SPDX-License-Identifier: AGPL-3.0-or-later
// Plays raw PCM to the default PulseAudio sink via ffplay. A writer thread
// feeds ffplay's stdin with NON-BLOCKING writes so stop() can never deadlock
// on a full pipe (a blocking fwrite there would hang thread::join()).
#pragma once
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>

class AudioPlayer {
public:
    AudioPlayer(const char* fmt = "s16le", int rate = 16000) : _fmt(fmt), _rate(rate) {}
    ~AudioPlayer() { stop(); }

    void start() {
        if (_running) { return; }
        char cmd[256];
        snprintf(cmd, sizeof(cmd),
                 "ffplay -f %s -ar %d -ch_layout mono -i pipe:0 -nodisp -autoexit -loglevel quiet",
                 _fmt, _rate);
        _pipe = popen(cmd, "w");
        if (!_pipe) { return; }
        _fd = fileno(_pipe);
        fcntl(_fd, F_SETFL, O_NONBLOCK);
        _stop = false;
        _running = true;
        _thread = std::thread(&AudioPlayer::run, this);
    }

    void stop() {
        if (!_running) { return; }
        _stop = true;
        _cv.notify_all();
        if (_thread.joinable()) { _thread.join(); }  // writer is non-blocking, exits promptly
        if (_pipe) { pclose(_pipe); _pipe = nullptr; }
        _running = false;
        std::lock_guard<std::mutex> l(_mtx);
        _q.clear();
    }

    void feed(const unsigned char* data, size_t len) {
        if (!_running) { return; }
        std::lock_guard<std::mutex> l(_mtx);
        _q.emplace_back(data, data + len);
        while (_q.size() > 4000) { _q.pop_front(); }
        _cv.notify_one();
    }

    bool running() const { return _running; }

private:
    void run() {
        while (!_stop) {
            std::vector<unsigned char> buf;
            {
                std::unique_lock<std::mutex> l(_mtx);
                _cv.wait_for(l, std::chrono::milliseconds(100),
                             [this] { return _stop || !_q.empty(); });
                if (_stop) { return; }
                if (_q.empty()) { continue; }
                buf = std::move(_q.front());
                _q.pop_front();
            }
            size_t off = 0;
            while (off < buf.size() && !_stop) {
                ssize_t n = write(_fd, buf.data() + off, buf.size() - off);
                if (n > 0) { off += (size_t)n; }
                else if (n < 0 && errno == EAGAIN) {
                    std::unique_lock<std::mutex> l(_mtx);
                    _cv.wait_for(l, std::chrono::milliseconds(50),
                                 [this] { return _stop.load(); });
                } else { return; }  // EPIPE / error: ffplay gone
            }
        }
    }

    FILE* _pipe = nullptr;
    int _fd = -1;
    const char* _fmt;
    int _rate;
    std::atomic<bool> _running{false};
    std::atomic<bool> _stop{false};
    std::thread _thread;
    std::mutex _mtx;
    std::condition_variable _cv;
    std::deque<std::vector<unsigned char>> _q;
};
