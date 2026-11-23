// SPDX-License-Identifier: AGPL-3.0-or-later
// Minimal libwebsockets client for the radio-translator daemon. Runs its own
// service thread; the audio thread only enqueues PCM (non-blocking). Receives
// JSON chunk/status events and forwards them via callbacks. Auto-reconnects.
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <libwebsockets.h>

class WsClient {
public:
    std::function<void(const std::string&)> onText;    // translated chunk text
    std::function<void(const std::string&)> onStatus;  // tier/model status line
    std::function<void(unsigned char, const unsigned char*, size_t)> onAudio;  // (type, PCM)

    WsClient();
    ~WsClient();
    void setUrl(const std::string& url);
    void start();
    void stop();
    void sendPcm(const float* data, int count);  // mono 16 kHz float
    void sendText(const std::string& msg);       // JSON control message
    bool connected() const { return _connected; }
    static int callback(struct lws* wsi, enum lws_callback_reasons reason,
                        void* user, void* in, size_t len);

private:
    void run();
    void doConnect();
    void handleReceive(const char* data, size_t len);
    void pumpSend(struct lws* wsi);

    std::string _host = "127.0.0.1", _path = "/";
    int _port = 8770;
    bool _ssl = false;
    std::atomic<bool> _stopping{false};
    std::atomic<bool> _connected{false};
    std::atomic<bool> _connecting{false};
    struct lws_context* _ctx = nullptr;
    struct lws* _wsi = nullptr;
    std::thread _thread;
    std::mutex _qMtx;
    std::deque<std::pair<std::vector<unsigned char>, bool>> _queue;  // (LWS_PRE+payload, isBinary)
    std::string _rxBuf;
};
