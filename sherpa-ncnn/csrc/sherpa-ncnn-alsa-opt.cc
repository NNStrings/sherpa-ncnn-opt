#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cctype>
#include <vector>
#include <sys/resource.h>

#include "sherpa-ncnn/csrc/alsa.h"
#include "sherpa-ncnn/csrc/display.h"
#include "sherpa-ncnn/csrc/recognizer.h"

bool stop = false;

static void Handler(int sig) {
  stop = true;
  fprintf(stderr, "\nExiting...\n");
}

// 预分配全局音频缓冲区
constexpr int32_t kBufferSize = 1600; // 0.1秒@16kHz
float g_audio_buffer[kBufferSize] = {0};

// 实时优先级设置
void SetRealtimePriority() {
  setpriority(PRIO_PROCESS, 0, -20); // 最高优先级
}

// 识别处理循环
void ProcessStream(sherpa_ncnn::Recognizer& recognizer, 
                   sherpa_ncnn::Alsa& alsa) {
  auto s = recognizer.CreateStream();
  sherpa_ncnn::Display display;
  int32_t segment_index = 0;
  std::string last_text;

  while (!stop) {
    const auto& samples = alsa.Read(kBufferSize);
    memcpy(g_audio_buffer, samples.data(), samples.size() * sizeof(float));

    s->AcceptWaveform(16000, g_audio_buffer, samples.size());

    while (recognizer.IsReady(s.get())) {
      recognizer.DecodeStream(s.get());
    }

    bool is_endpoint = recognizer.IsEndpoint(s.get());
    auto text = recognizer.GetResult(s.get()).text;

    if (!text.empty() && last_text != text) {
      std::transform(text.begin(), text.end(), text.begin(), ::tolower);
      display.Print(segment_index, text);
      last_text = text;
    }

    if (is_endpoint) {
      s->Finalize(); // 通知识别器该段落结束

      if (!text.empty()) ++segment_index;

      recognizer.Reset(s.get()); // 重置流对象以继续
      last_text.clear();         // 清除上一段文本，避免重复显示
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 9) {
    fprintf(stderr, "Usage: %s tokens.txt encoder.param encoder.bin decoder.param decoder.bin joiner.param joiner.bin device_name\n", argv[0]);
    return EXIT_FAILURE;
  }

  signal(SIGINT, Handler);
  SetRealtimePriority();

  sherpa_ncnn::RecognizerConfig config;
  config.model_config.tokens = argv[1];
  config.model_config.encoder_param = argv[2];
  config.model_config.encoder_bin = argv[3];
  config.model_config.decoder_param = argv[4];
  config.model_config.decoder_bin = argv[5];
  config.model_config.joiner_param = argv[6];
  config.model_config.joiner_bin = argv[7];

  config.model_config.encoder_opt.num_threads = 1;
  config.model_config.decoder_opt.num_threads = 1;
  config.model_config.joiner_opt.num_threads = 1;

  // 启用端点检测
  config.enable_endpoint = true;
  config.endpoint_config.rule1.min_trailing_silence = 2.4f;
  config.endpoint_config.rule2.min_trailing_silence = 1.2f;
  config.endpoint_config.rule3.min_utterance_length = 300.0f;

  config.feat_config.sampling_rate = 16000;
  config.feat_config.feature_dim = 80;

  sherpa_ncnn::Alsa alsa(argv[8]);
  if (alsa.GetExpectedSampleRate() != 16000) {
    fprintf(stderr, "Unsupported sample rate: %d\n", alsa.GetExpectedSampleRate());
    return EXIT_FAILURE;
  }

  sherpa_ncnn::Recognizer recognizer(config);
  ProcessStream(recognizer, alsa);

  return EXIT_SUCCESS;
}
