// SPDX-License-Identifier: AGPL-3.0-or-later
#include "ws_client.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstring>

using json = nlohmann::json;

static const struct lws_protocols protocols[] = {
    { "radio-translator", WsClient::callback, 0, 4096, 0, NULL, 0 },
    { NULL, NULL, 0, 0, 0, NULL, 0 }
};

WsClient::WsClient() {}
WsClient::~WsClient() { stop(); }

void WsClient::setUrl(const std::string& url) {
    // ws://host:port/path  or  wss://host:port/path
    _ssl = url.rfind("wss://", 0) == 0;
    std::string rest = url.substr(url.find("://") + 3);
    std::string hostport = rest.substr(0, rest.find('/'));
    _path = rest.find('/') == std::string::npos ? "/" : rest.substr(rest.find('/'));
    auto colon = hostport.find(':');
    if (colon == std::string::npos) {
        _host = hostport;
        _port = _ssl ? 443 : 80;
    } else {
        _host = hostport.substr(0, colon);
        _port = std::stoi(hostport.substr(colon + 1));
    }
}

void WsClient::start() {
    if (_thread.joinable()) { return; }
    _stopping = false;
    _thread = std::thread(&WsClient::run, this);
}

void WsClient::stop() {
    _stopping = true;
    if (_ctx && _connected) { lws_cancel_service(_ctx); }
    if (_thread.joinable()) { _thread.join(); }
}

void WsClient::sendPcm(const float* data, int count) {
    if (count <= 0) { return; }
    std::vector<unsigned char> buf(LWS_PRE + (size_t)count * 2);
    auto* out = reinterpret_cast<int16_t*>(buf.data() + LWS_PRE);
    for (int i = 0; i < count; i++) {
        float v = data[i];
        if (v > 1.0f) { v = 1.0f; } else if (v < -1.0f) { v = -1.0f; }
        out[i] = (int16_t)(v * 32767.0f);
    }
    {
        std::lock_guard<std::mutex> l(_qMtx);
        _queue.push_back({std::move(buf), true});
        // bound the queue so a stalled daemon cannot grow memory without limit
        while (_queue.size() > 200) { _queue.pop_front(); }
    }
    if (_ctx && _connected) { lws_cancel_service(_ctx); }
}

void WsClient::sendText(const std::string& msg) {
    std::vector<unsigned char> buf(LWS_PRE + msg.size());
    memcpy(buf.data() + LWS_PRE, msg.data(), msg.size());
    {
        std::lock_guard<std::mutex> l(_qMtx);
        _queue.push_back({std::move(buf), false});
    }
    if (_ctx && _connected) { lws_cancel_service(_ctx); }
}

void WsClient::doConnect() {
    _connecting = true;
    struct lws_client_connect_info i;
    memset(&i, 0, sizeof(i));
    i.context = _ctx;
    i.address = _host.c_str();
    i.port = _port;
    i.path = _path.c_str();
    i.host = i.address;
    i.origin = i.address;
    i.protocol = protocols[0].name;
    i.local_protocol_name = protocols[0].name;
    i.ssl_connection = _ssl ? LCCSCF_USE_SSL : 0;
    if (!lws_client_connect_via_info(&i)) {
        _connecting = false;
    }
}

void WsClient::run() {
    struct lws_context_creation_info info;
    memset(&info, 0, sizeof(info));
    info.port = CONTEXT_PORT_NO_LISTEN;
    info.protocols = protocols;
    info.user = this;
    info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
    info.timeout_secs = 10;
    _ctx = lws_create_context(&info);
    if (!_ctx) { return; }

    auto lastAttempt = std::chrono::steady_clock::now() - std::chrono::seconds(2);
    while (!_stopping) {
        if (!_connected && !_connecting) {
            auto now = std::chrono::steady_clock::now();
            if (now - lastAttempt > std::chrono::seconds(1)) {
                lastAttempt = now;
                doConnect();
            }
        }
        lws_service(_ctx, 50);
        if (_connected && _wsi) {
            std::lock_guard<std::mutex> l(_qMtx);
            if (!_queue.empty()) { lws_callback_on_writable(_wsi); }
        }
    }
    lws_context_destroy(_ctx);
    _ctx = nullptr;
}

void WsClient::pumpSend(struct lws* wsi) {
    std::vector<unsigned char> buf;
    bool isBinary = true;
    {
        std::lock_guard<std::mutex> l(_qMtx);
        if (_queue.empty()) { return; }
        buf = std::move(_queue.front().first);
        isBinary = _queue.front().second;
        _queue.pop_front();
    }
    lws_write(wsi, buf.data() + LWS_PRE, buf.size() - LWS_PRE,
              isBinary ? LWS_WRITE_BINARY : LWS_WRITE_TEXT);
    std::lock_guard<std::mutex> l(_qMtx);
    if (!_queue.empty()) { lws_callback_on_writable(wsi); }
}

void WsClient::handleReceive(const char* data, size_t len) {
    json ev;
    try { ev = json::parse(std::string(data, len)); }
    catch (...) { return; }
    std::string type = ev.value("type", "");
    if (type == "chunk") {
        std::string text = ev.value("text", "");
        bool sup = ev.value("suppressed", false);
        if (onText) { onText(sup ? "[!] " + text : text); }
        if (onStatus) {
            char line[160];
            snprintf(line, sizeof(line), "%s  %.1fs  chunk %d%s",
                     ev.value("language", "?").c_str(),
                     ev.value("latency_s", 0.0),
                     ev.value("chunk", 0), sup ? "  (suppressed)" : "");
            onStatus(line);
        }
    } else if (type == "status") {
        int pv = ev.value("protocol_version", 0);
        std::string line = ev.value("tier", "?") + " / " + ev.value("model", "?") +
                           " / " + ev.value("device", "?");
        if (pv != 1) {
            line += "  [protocol mismatch: daemon v" + std::to_string(pv) + ", module v1]";
        }
        if (onStatus) { onStatus(line); }
    }
}

int WsClient::callback(struct lws* wsi, enum lws_callback_reasons reason,
                       void* user, void* in, size_t len) {
    auto* self = static_cast<WsClient*>(lws_context_user(lws_get_context(wsi)));
    if (!self) { return 0; }
    switch (reason) {
        case LWS_CALLBACK_CLIENT_ESTABLISHED:
            self->_connected = true;
            self->_connecting = false;
            self->_wsi = wsi;
            lws_callback_on_writable(wsi);
            break;
        case LWS_CALLBACK_CLIENT_RECEIVE:
            if (lws_frame_is_binary(wsi)) {
                if (self->onAudio && len >= 1) {
                    const unsigned char* p = static_cast<const unsigned char*>(in);
                    self->onAudio(p[0], p + 1, len - 1);  // p[0]: 0=RF audio, 1=TTS
                }
            } else {
                self->_rxBuf.append(static_cast<const char*>(in), len);
                if (lws_is_final_fragment(wsi)) {
                    self->handleReceive(self->_rxBuf.data(), self->_rxBuf.size());
                    self->_rxBuf.clear();
                }
            }
            break;
        case LWS_CALLBACK_CLIENT_WRITEABLE:
            self->pumpSend(wsi);
            break;
        case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
        case LWS_CALLBACK_CLOSED:
        case LWS_CALLBACK_WSI_DESTROY:
            self->_connected = false;
            self->_connecting = false;
            self->_wsi = nullptr;
            break;
        default:
            break;
    }
    return 0;
}
