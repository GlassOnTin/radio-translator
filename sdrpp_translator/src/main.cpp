// SPDX-License-Identifier: AGPL-3.0-or-later
// SDR++ translator module: taps a demod audio sink stream (default "Radio"),
// resamples to 16 kHz mono, streams it to the radio-translator daemon over
// WebSocket, and shows the returned English translation in a sidebar panel.
// The module runs no models; compute lives in the daemon (local or remote).
#include <module.h>
#include <core.h>
#include <config.h>
#include <gui/gui.h>
#include <gui/style.h>
#include <signal_path/signal_path.h>
#include <signal_path/sink.h>
#include <dsp/convert/stereo_to_mono.h>
#include <dsp/multirate/rational_resampler.h>
#include <dsp/sink/handler_sink.h>
#include <utils/event.h>
#include <json.hpp>
#include <mutex>
#include <string>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>
#include "ws_client.h"
#include "audio_player.h"

#define CONCAT(a, b) ((std::string(a) + b).c_str())

SDRPP_MOD_INFO{
    /* Name:         */ "translator",
    /* Description:*/ "Live audio translation via radio-translator daemon",
    /* Author:     */ "radio-translator",
    /* Version:    */ 0, 1, 0,
    /* Max instances*/ -1
};

ConfigManager config;

class TranslatorModule : public ModuleManager::Instance {
public:
    std::string name;
    bool enabled = true;
    std::string url = "ws://127.0.0.1:8770";
    std::string streamName = "Radio";
    std::string lang = "";
    bool useIsolation = true;
    bool playAudio = false;
    bool playTts = false;
    float rfVolume = 1.0f;
    float ttsVolume = 1.0f;
    std::string voice = "af_heart";
    float ttsSpeed = 1.0f;
    bool gates = true;
    std::string backend = "whisper";
    float chunkSeconds = 12.0f;
    int silenceMs = 300;
    float bufferMinutes = 10.0f;

    TranslatorModule(std::string name) : name(name) {
        config.acquire();
        if (config.conf[name].contains("url")) { url = config.conf[name]["url"].get<std::string>(); }
        if (config.conf[name].contains("streamName")) { streamName = config.conf[name]["streamName"].get<std::string>(); }
        if (config.conf[name].contains("lang")) { lang = config.conf[name]["lang"].get<std::string>(); }
        if (config.conf[name].contains("useIsolation")) { useIsolation = config.conf[name]["useIsolation"].get<bool>(); }
        if (config.conf[name].contains("playAudio")) { playAudio = config.conf[name]["playAudio"].get<bool>(); }
        if (config.conf[name].contains("playTts")) { playTts = config.conf[name]["playTts"].get<bool>(); }
        if (config.conf[name].contains("rfVolume")) { rfVolume = config.conf[name]["rfVolume"].get<float>(); }
        if (config.conf[name].contains("ttsVolume")) { ttsVolume = config.conf[name]["ttsVolume"].get<float>(); }
        if (config.conf[name].contains("voice")) { voice = config.conf[name]["voice"].get<std::string>(); }
        if (config.conf[name].contains("ttsSpeed")) { ttsSpeed = config.conf[name]["ttsSpeed"].get<float>(); }
        if (config.conf[name].contains("gates")) { gates = config.conf[name]["gates"].get<bool>(); }
        if (config.conf[name].contains("chunkSeconds")) { chunkSeconds = config.conf[name]["chunkSeconds"].get<float>(); }
        if (config.conf[name].contains("silenceMs")) { silenceMs = config.conf[name]["silenceMs"].get<int>(); }
        if (config.conf[name].contains("bufferMinutes")) { bufferMinutes = config.conf[name]["bufferMinutes"].get<float>(); }
        config.release();

        ws.onText = [this](const std::string& t) {
            std::lock_guard<std::mutex> l(textMtx);
            englishText = t;
            double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            transcriptEntries.push_back({secs, t});
            double cutoff = secs - bufferMinutes * 60.0;
            while (!transcriptEntries.empty() && transcriptEntries.front().first < cutoff) {
                transcriptEntries.pop_front();
            }
            transcript.clear();
            double prev = 0.0;
            for (const auto& e : transcriptEntries) {
                if (prev > 0 && e.first - prev > 3.0) { transcript += "\n\n"; }
                else if (!transcript.empty()) { transcript += " "; }
                transcript += e.second;
                prev = e.first;
            }
        };
        ws.onStatus = [this](const std::string& s) {
            std::lock_guard<std::mutex> l(textMtx); statusLine = s;
        };
        ws.onAudio = [this](unsigned char type, const unsigned char* d, size_t n) {
            if (type == 0 && playAudio) {
                std::vector<int16_t> s(n / 2);
                memcpy(s.data(), d, n);
                if (rfVolume < 0.999f) { for (auto& v : s) { v = (int16_t)(v * rfVolume); } }
                rfPlayer.feed(reinterpret_cast<const unsigned char*>(s.data()), s.size() * 2);
            } else if (type == 1 && playTts) {
                std::vector<float> s(n / 4);
                memcpy(s.data(), d, n);
                if (ttsVolume < 0.999f) { for (auto& v : s) { v *= ttsVolume; } }
                ttsPlayer.feed(reinterpret_cast<const unsigned char*>(s.data()), s.size() * 4);
            }
        };
        ws.setUrl(url);
        ws.start();
        // sync processing/output toggles to the daemon (queued until connected)
        ws.sendText(std::string("{\"enhance\":") + (useIsolation ? "\"isolate\"" : "\"none\"") + "}");
        ws.sendText("{\"chunk\":" + std::to_string(chunkSeconds) +
                    ",\"silence\":" + std::to_string(silenceMs) + "}");
        ws.sendText("{\"lang\":\"" + lang + "\",\"voice\":\"" + voice +
                    "\",\"speed\":" + std::to_string(ttsSpeed) +
                    ",\"gates\":" + (gates ? "true" : "false") + "}");
        if (playAudio) { rfPlayer.start(); ws.sendText("{\"audio\":true}"); }
        if (playTts) { ttsPlayer.start(); ws.sendText("{\"tts\":true}"); }
        gui::menu.registerEntry(name, menuHandler, this, this);
    }

    ~TranslatorModule() {
        gui::menu.removeEntry(name);
        untap();
        rfPlayer.stop();
        ttsPlayer.stop();
        sigpath::sinkManager.onStreamRegistered.unbindHandler(&streamRegHandler);
        ws.stop();
    }

    void postInit() {
        tryTap();
        streamRegHandler.ctx = this;
        streamRegHandler.handler = [](std::string n, void* ctx) {
            auto* _this = (TranslatorModule*)ctx;
            if (n == _this->streamName) { _this->tryTap(); }
        };
        sigpath::sinkManager.onStreamRegistered.bindHandler(&streamRegHandler);
    }

    void enable() { enabled = true; tryTap(); }
    void disable() { enabled = false; untap(); }
    bool isEnabled() { return enabled; }

private:
    dsp::stream<dsp::stereo_t>* audio = nullptr;
    dsp::convert::StereoToMono s2m;
    dsp::multirate::RationalResampler<float> resamp;
    dsp::sink::Handler<float> tap;
    WsClient ws;
    AudioPlayer rfPlayer{"s16le", 16000};
    AudioPlayer ttsPlayer{"f32le", 24000};
    bool tapped = false;
    float tappedRate = 0.0f;
    std::mutex textMtx;
    std::string englishText = "(waiting for translation...)";
    std::string statusLine = "disconnected";
    std::string transcript;
    std::deque<std::pair<double, std::string>> transcriptEntries;
    bool configDirty = false;
    double lastSaveTime = 0.0;
    EventHandler<std::string> streamRegHandler;

    void tryTap() {
        if (tapped || !enabled) { return; }
        audio = sigpath::sinkManager.bindStream(streamName);
        if (!audio) { return; }
        tappedRate = sigpath::sinkManager.getStreamSampleRate(streamName);
        s2m.init(audio);
        resamp.init(&s2m.out, tappedRate, 16000.0);
        tap.init(&resamp.out, pcmHandler, this);
        s2m.start();
        resamp.start();
        tap.start();
        tapped = true;
    }

    void untap() {
        if (!tapped) { return; }
        tap.stop();
        resamp.stop();
        s2m.stop();
        sigpath::sinkManager.unbindStream(streamName, audio);
        audio = nullptr;
        tapped = false;
    }

    // coalesce config writes: sliders fire every frame, so just mark dirty and
    // flush at most once a second (avoids hammering the shared config mutex).
    void saveConfig() { configDirty = true; }

    void writeConfigNow() {
        config.acquire();
        config.conf[name]["url"] = url;
        config.conf[name]["streamName"] = streamName;
        config.conf[name]["lang"] = lang;
        config.conf[name]["useIsolation"] = useIsolation;
        config.conf[name]["playAudio"] = playAudio;
        config.conf[name]["playTts"] = playTts;
        config.conf[name]["rfVolume"] = rfVolume;
        config.conf[name]["ttsVolume"] = ttsVolume;
        config.conf[name]["voice"] = voice;
        config.conf[name]["ttsSpeed"] = ttsSpeed;
        config.conf[name]["gates"] = gates;
        config.conf[name]["chunkSeconds"] = chunkSeconds;
        config.conf[name]["silenceMs"] = silenceMs;
        config.conf[name]["bufferMinutes"] = bufferMinutes;
        config.release(true);
    }

    static void pcmHandler(float* data, int count, void* ctx) {
        auto* _this = (TranslatorModule*)ctx;
        if (_this->enabled) { _this->ws.sendPcm(data, count); }
    }

    static void menuHandler(void* ctx) {
        auto* _this = (TranslatorModule*)ctx;
        float menuWidth = ImGui::GetContentRegionAvail().x;
        if (!_this->enabled) { style::beginDisabled(); }

        // follow AF sample-rate changes without re-tapping
        if (_this->tapped) {
            float sr = sigpath::sinkManager.getStreamSampleRate(_this->streamName);
            if (sr > 0 && sr != _this->tappedRate) {
                _this->resamp.setInSamplerate(sr);
                _this->tappedRate = sr;
            }
        }

        ImGui::SetNextItemWidth(menuWidth);
        char urlBuf[256];
        strncpy(urlBuf, _this->url.c_str(), sizeof(urlBuf) - 1);
        urlBuf[sizeof(urlBuf) - 1] = 0;
        if (ImGui::InputText(CONCAT("Daemon URL##rt_", _this->name), urlBuf, sizeof(urlBuf))) {
            _this->url = urlBuf;
            _this->saveConfig();
        }
        if (ImGui::Button(CONCAT("Reconnect##rt_", _this->name), ImVec2(menuWidth, 0))) {
            _this->ws.stop();
            _this->ws.setUrl(_this->url);
            _this->ws.start();
        }

        auto names = sigpath::sinkManager.getStreamNames();
        if (ImGui::BeginCombo(CONCAT("Audio stream##rt_", _this->name), _this->streamName.c_str())) {
            for (auto& n : names) {
                if (ImGui::Selectable(n.c_str(), n == _this->streamName)) {
                    _this->streamName = n;
                    _this->saveConfig();
                    _this->untap();
                    _this->tryTap();
                }
            }
            ImGui::EndCombo();
        }

        if (ImGui::Checkbox(CONCAT("Voice isolation##rt_", _this->name), &_this->useIsolation)) {
            _this->ws.sendText(std::string("{\"enhance\":") +
                               (_this->useIsolation ? "\"isolate\"" : "\"none\"") + "}");
            _this->saveConfig();
        }
        if (ImGui::Checkbox(CONCAT("Play filtered audio##rt_", _this->name), &_this->playAudio)) {
            if (_this->playAudio) {
                _this->rfPlayer.start();
                _this->ws.sendText("{\"audio\":true}");
            } else {
                _this->rfPlayer.stop();
                _this->ws.sendText("{\"audio\":false}");
            }
            _this->saveConfig();
        }
        if (ImGui::Checkbox(CONCAT("English speech (TTS)##rt_", _this->name), &_this->playTts)) {
            if (_this->playTts) {
                _this->ttsPlayer.start();
                _this->ws.sendText("{\"tts\":true}");
            } else {
                _this->ttsPlayer.stop();
                _this->ws.sendText("{\"tts\":false}");
            }
            _this->saveConfig();
        }
        if (ImGui::SliderFloat(CONCAT("Filtered audio volume##rt_", _this->name),
                               &_this->rfVolume, 0.0f, 1.0f, "%.2f")) {
            _this->saveConfig();
        }
        if (ImGui::SliderFloat(CONCAT("English speech volume##rt_", _this->name),
                               &_this->ttsVolume, 0.0f, 1.0f, "%.2f")) {
            _this->saveConfig();
        }
        {
            static const char* langs[] = {"auto","zh","en","fr","de","es","ru","ar",
                                          "ja","ko","it","pt","tr","uk","el","fa"};
            int idx = 0;
            for (int i = 0; i < 16; i++) { if (_this->lang == langs[i]) { idx = i; } }
            if (ImGui::Combo(CONCAT("Source language##rt_", _this->name), &idx, langs, 16)) {
                _this->lang = (idx == 0) ? "" : langs[idx];
                _this->ws.sendText("{\"lang\":\"" + _this->lang + "\"}");
                _this->saveConfig();
            }
        }
        {
            static const char* voices[] = {"af_heart","af_sky","af_bella","af_nova","af_sarah",
                                           "am_adam","am_michael","am_onyx","bf_emma","bf_alice",
                                           "bm_george","bm_fable"};
            int idx = 0;
            for (int i = 0; i < 12; i++) { if (_this->voice == voices[i]) { idx = i; } }
            if (ImGui::Combo(CONCAT("English voice##rt_", _this->name), &idx, voices, 12)) {
                _this->voice = voices[idx];
                _this->ws.sendText("{\"voice\":\"" + _this->voice + "\"}");
                _this->saveConfig();
            }
        }
        if (ImGui::SliderFloat(CONCAT("Speech speed##rt_", _this->name),
                               &_this->ttsSpeed, 0.5f, 2.0f, "%.2f")) {
            _this->ws.sendText("{\"speed\":" + std::to_string(_this->ttsSpeed) + "}");
            _this->saveConfig();
        }
        if (ImGui::Checkbox(CONCAT("Suppress hallucinations##rt_", _this->name), &_this->gates)) {
            _this->ws.sendText(std::string("{\"gates\":") + (_this->gates ? "true" : "false") + "}");
            _this->saveConfig();
        }
        {
            static const char* backs[] = {"whisper", "whisper-mt", "canary", "seamless"};
            int idx = 0;
            for (int i = 0; i < 4; i++) { if (_this->backend == backs[i]) { idx = i; } }
            if (ImGui::Combo(CONCAT("Backend##rt_", _this->name), &idx, backs, 4)) {
                _this->backend = backs[idx];
                _this->ws.sendText("{\"backend\":\"" + _this->backend + "\"}");
            }
            ImGui::TextUnformatted("seamless = non-commercial S2ST (opt-in)");
        }

        ImGui::TextUnformatted(_this->ws.connected() ? "connected" : "disconnected");
        {
            std::lock_guard<std::mutex> l(_this->textMtx);
            ImGui::TextWrapped("%s", _this->statusLine.c_str());
        }
        if (ImGui::SliderFloat(CONCAT("Chunk seconds##rt_", _this->name),
                               &_this->chunkSeconds, 3.0f, 30.0f, "%.1f")) {
            _this->ws.sendText("{\"chunk\":" + std::to_string(_this->chunkSeconds) + "}");
            _this->saveConfig();
        }
        if (ImGui::SliderInt(CONCAT("Silence ms##rt_", _this->name),
                             &_this->silenceMs, 100, 1000)) {
            _this->ws.sendText("{\"silence\":" + std::to_string(_this->silenceMs) + "}");
            _this->saveConfig();
        }
        if (ImGui::SliderFloat(CONCAT("Transcript minutes##rt_", _this->name),
                               &_this->bufferMinutes, 1.0f, 60.0f, "%.0f")) {
            _this->saveConfig();
        }
        if (ImGui::Button(CONCAT("Copy transcript##rt_", _this->name), ImVec2(menuWidth, 0))) {
            std::lock_guard<std::mutex> l(_this->textMtx);
            ImGui::SetClipboardText(_this->transcript.c_str());
        }
        ImGui::Separator();
        ImGui::BeginChild(CONCAT("transcript##rt_", _this->name), ImVec2(0, 220), true);
        {
            std::lock_guard<std::mutex> l(_this->textMtx);
            // render only the recent tail: word-wrapping the whole 10-minute
            // buffer every frame tanks the GUI frame rate (laggy toggles).
            // The full transcript is still kept for the Copy button.
            const std::string& full = _this->transcript;
            size_t off = full.size() > 3000 ? full.size() - 3000 : 0;
            while (off > 0 && full[off] != ' ' && full[off] != '\n') { off++; }
            ImGui::TextWrapped("%s", full.c_str() + off);
            if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 24.0f) {
                ImGui::SetScrollHereY(1.0f);
            }
        }
        ImGui::EndChild();

        double now = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (_this->configDirty && now - _this->lastSaveTime > 1.0) {
            _this->writeConfigNow();
            _this->configDirty = false;
            _this->lastSaveTime = now;
        }

        if (!_this->enabled) { style::endDisabled(); }
    }
};

MOD_EXPORT void _INIT_() {
    config.setPath(core::args["root"].s() + "/translator_config.json");
    config.load(nlohmann::json({}));
    config.enableAutoSave();
}

MOD_EXPORT ModuleManager::Instance* _CREATE_INSTANCE_(std::string name) {
    return new TranslatorModule(name);
}

MOD_EXPORT void _DELETE_INSTANCE_(void* instance) {
    delete (TranslatorModule*)instance;
}

MOD_EXPORT void _END_() {
    config.disableAutoSave();
    config.save();
}
