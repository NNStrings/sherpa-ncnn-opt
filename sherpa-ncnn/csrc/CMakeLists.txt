include_directories(${CMAKE_SOURCE_DIR})

set(sherpa_ncnn_core_srcs
  context-graph.cc
  conv-emformer-model.cc
  decoder.cc
  endpoint.cc
  features.cc
  file-utils.cc
  greedy-search-decoder.cc
  hypothesis.cc
  lstm-model.cc
  meta-data.cc
  model.cc
  modified-beam-search-decoder.cc
  poolingmodulenoproj.cc
  recognizer.cc
  resample.cc
  simpleupsample.cc
  stack.cc
  stream.cc
  symbol-table.cc
  tensorasstrided.cc
  wave-reader.cc
  wave-writer.cc
  zipformer-model.cc
)

list(APPEND sherpa_ncnn_core_srcs
  circular-buffer.cc
  silero-vad-model-config.cc
  silero-vad-model.cc
  voice-activity-detector.cc
)

add_library(sherpa-ncnn-core ${sherpa_ncnn_core_srcs})
target_link_libraries(sherpa-ncnn-core PUBLIC kaldi-native-fbank-core ncnn)

if(SHERPA_NCNN_ENABLE_PYTHON AND WIN32)
  install(TARGETS sherpa-ncnn-core DESTINATION ..)
else()
  install(TARGETS sherpa-ncnn-core DESTINATION lib)
endif()

if(NOT SHERPA_NCNN_ENABLE_PYTHON)
  if(SHERPA_NCNN_ENABLE_BINARY)
    add_executable(sherpa-ncnn sherpa-ncnn.cc)
    target_link_libraries(sherpa-ncnn PRIVATE sherpa-ncnn-core)
    install(TARGETS sherpa-ncnn DESTINATION bin)

    add_executable(sherpa-ncnn-opt sherpa-ncnn-opt.cc)
    target_link_libraries(sherpa-ncnn-opt PRIVATE sherpa-ncnn-core)
    find_library(PROFILER_LIB profiler)
    find_library(TCMALLOC_LIB tcmalloc)
    find_package(Threads REQUIRED)
    target_link_libraries(sherpa-ncnn-opt PRIVATE Threads::Threads ${PROFILER_LIB} ${TCMALLOC_LIB})
    install(TARGETS sherpa-ncnn-opt DESTINATION bin)

    add_executable(sherpa-ncnn-vad sherpa-ncnn-vad.cc)
    target_link_libraries(sherpa-ncnn-vad PRIVATE sherpa-ncnn-core)
    install(TARGETS sherpa-ncnn-vad DESTINATION bin)

    if(SHERPA_NCNN_HAS_ALSA)
      add_executable(sherpa-ncnn-alsa sherpa-ncnn-alsa.cc alsa.cc)
      add_executable(sherpa-ncnn-alsa-opt sherpa-ncnn-alsa-opt.cc alsa.cc)
      target_link_libraries(sherpa-ncnn-alsa PRIVATE sherpa-ncnn-core)
      target_link_libraries(sherpa-ncnn-alsa-opt PRIVATE sherpa-ncnn-core)

      if(DEFINED ENV{SHERPA_NCNN_ALSA_LIB_DIR})
        target_link_libraries(sherpa-ncnn-alsa PRIVATE -L$ENV{SHERPA_NCNN_ALSA_LIB_DIR} -lasound)
        target_link_libraries(sherpa-ncnn-alsa-opt PRIVATE -L$ENV{SHERPA_NCNN_ALSA_LIB_DIR} -lasound)
      else()
        target_link_libraries(sherpa-ncnn-alsa PRIVATE asound)
        target_link_libraries(sherpa-ncnn-alsa-opt PRIVATE asound)
      endif()
      install(TARGETS sherpa-ncnn-alsa DESTINATION bin)
      install(TARGETS sherpa-ncnn-alsa-opt DESTINATION bin)
    endif()

    if(SHERPA_NCNN_ENABLE_PORTAUDIO)
      add_executable(sherpa-ncnn-microphone
        sherpa-ncnn-microphone.cc
        microphone.cc
      )

      if(BUILD_SHARED_LIBS)
        set(PA_LIB portaudio)
      else()
        set(PA_LIB portaudio_static)
      endif()

      target_link_libraries(sherpa-ncnn-microphone PRIVATE ${PA_LIB} sherpa-ncnn-core)

      install(TARGETS sherpa-ncnn-microphone DESTINATION bin)
    endif()

    if(SHERPA_NCNN_ENABLE_GENERATE_INT8_SCALE_TABLE)
      add_executable(generate-int8-scale-table generate-int8-scale-table.cc)
      target_link_libraries(generate-int8-scale-table sherpa-ncnn-core)
    endif()
  endif()
endif()

if(SHERPA_NCNN_ENABLE_TEST)
  add_executable(test-resample test-resample.cc)
  target_link_libraries(test-resample sherpa-ncnn-core)
  add_executable(test-context-graph test-context-graph.cc)
  target_link_libraries(test-context-graph sherpa-ncnn-core)
endif()
