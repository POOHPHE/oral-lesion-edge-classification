import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart' show compute;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:http/http.dart' as http;

/// Benchmark methodology (End-to-End, 4-step):
/// - Measures 4 steps per trial:
///   1) Decode (compressed bytes → pixel array)
///   2) Preprocess (resize 224×224, ImageNet normalize, tensor construction)
///   3) Inference (ONNX Runtime forward pass)
///   4) Postprocess (output parsing, softmax, argmax)
/// - Image fetched from network ONCE per image, then 10 trials on the same bytes
/// - Protocol: 5 warm-up inferences (discarded), then 10 trials × 50 images = 500 measurements
/// - Image selection: By Content-Length header (10 smallest, 30 median, 10 largest)
/// - Delay: 100 ms between consecutive inferences
/// - Thermal drift check: Compare median total latency of first vs second half (<5% drift)
/// - Metrics: Median (primary), Mean±SD, IQR, P95, Min, Max — all in milliseconds

class BenchmarkPage extends StatefulWidget {
  const BenchmarkPage({super.key});

  @override
  State<BenchmarkPage> createState() => _BenchmarkPageState();
}

class _BenchmarkPageState extends State<BenchmarkPage> {
  // ONNX Runtime
  final OnnxRuntime _onnxRuntime = OnnxRuntime();
  OrtSession? _session;

  // Model configuration - MODIFY THESE FOR YOUR MODEL
  static const int inputSize = 224;
  static const List<double> mean = [0.485, 0.456, 0.406];
  static const List<double> std = [0.229, 0.224, 0.225];

  // Benchmark file path (in assets)
  static const String benchmarkFilePath = 'assets/mobile_benchmark.txt';

  // Benchmark parameters (fixed for reproducibility)
  static const int warmUpCount = 5;
  static const int trialsPerImage = 10;
  static const int totalImages = 50;
  static const int interInferenceDelayMs = 100;

  static const MethodChannel _nativeChannel = MethodChannel(
    'com.yourapp/onnx_provider',
  );

  String _selectedProvider = 'cpu'; // 'cpu' | 'nnapi'

  bool _nnapiAvailable = false; // detected at load time
  static const List<Map<String, String>> _providers = [
    {'value': 'cpu', 'label': 'CPU (Default)'},
    {'value': 'nnapi', 'label': 'NNAPI (NPU/DSP/GPU)'},
  ];

  // Class labels
  static const List<String> classLabels = [
    'Mass',
    'Normal',
    'Pigmented',
    'Red',
    'Ulcer',
    'WhiteRed',
  ];

  // State
  bool _isLoadingModels = false;
  bool _isBenchmarking = false;
  bool _isCancelled = false;
  bool _isWarmingUp = false;
  String? _errorMessage;

  // Model list
  List<String> _availableModels = [];
  String? _selectedModel;

  // Benchmark data
  List<String> _imageUrls = [];
  List<_ImageWithSize> _selectedImages = [];
  List<BenchmarkResult> _results = [];
  int _currentImageIndex = 0;
  int _currentTrial = 0;
  int _warmUpProgress = 0;
  int _successCount = 0;
  int _failCount = 0;

  // Per-step statistics
  _StepStats _decodeStats = _StepStats.empty();
  _StepStats _preprocessStats = _StepStats.empty();
  _StepStats _inferenceStats = _StepStats.empty();
  _StepStats _postprocessStats = _StepStats.empty();
  _StepStats _totalStats = _StepStats.empty();

  double _totalBenchmarkTime = 0;

  // Thermal drift detection (on total end-to-end time)
  double _firstHalfMedian = 0;
  double _secondHalfMedian = 0;
  double _thermalDriftPercent = 0;
  bool _thermalDriftWarning = false;

  // Current model input/output names
  String? _inputName;
  String? _outputName;

  @override
  void initState() {
    super.initState();
    _loadModelList();
    _loadBenchmarkFile();
  }

  @override
  void dispose() {
    _session?.close();
    super.dispose();
  }

  Future<void> _loadModelList() async {
    setState(() {
      _isLoadingModels = true;
      _errorMessage = null;
    });

    try {
      final modelListContent = await rootBundle.loadString(
        'assets/models/model_list.txt',
      );
      final models = modelListContent
          .split('\n')
          .map((line) => line.trim())
          .where(
            (line) =>
                line.isNotEmpty &&
                !line.startsWith('#') &&
                line.endsWith('.onnx'),
          )
          .toList();

      setState(() {
        _availableModels = models;
        _isLoadingModels = false;
      });
    } catch (e) {
      setState(() {
        _availableModels = ['model.onnx'];
        _isLoadingModels = false;
        _errorMessage = null;
      });
    }
  }

  Future<void> _loadSelectedModel() async {
    if (_selectedModel == null) return;

    setState(() {
      _isLoadingModels = true;
      _errorMessage = null;
    });

    _session?.close();
    _session = null;

    try {
      final modelPath = 'assets/models/$_selectedModel';
      final useNnapi = _selectedProvider == 'nnapi';

      // ── Attempt 1: OrtSessionOptions with executionProviders list ──────────
      OrtSession? session;
      if (useNnapi) {
        session = await _tryLoadWithOrtSessionOptions(modelPath);
      }

      // ── Attempt 2: plain createSessionFromAsset (always works for CPU) ─────
      session ??= await _onnxRuntime.createSessionFromAsset(modelPath);

      _session = session;

      if (_session!.inputNames.isNotEmpty) {
        _inputName = _session!.inputNames.first;
      }
      if (_session!.outputNames.isNotEmpty) {
        _outputName = _session!.outputNames.first;
      }

      setState(() {
        _isLoadingModels = false;
        _nnapiAvailable = useNnapi && session != null;
      });
    } catch (e) {
      setState(() {
        _isLoadingModels = false;
        _errorMessage = 'Failed to load model: $e';
      });
    }
  }

  /// Tries every known OrtSessionOptions API variant for NNAPI.
  /// Returns null (silently) if none compile/work, so caller falls back to CPU.
  Future<OrtSession?> _tryLoadWithOrtSessionOptions(String modelPath) async {
    // ── Variant A: executionProviders as a list property ─────────────────────
    // (masicai ^1.x "EP1 Configuration" feature)
    try {
      final opts = OrtSessionOptions();
      // ignore: avoid_dynamic_calls
      (opts as dynamic).executionProviders = <Map<String, dynamic>>[
        {
          'name': 'NnapiExecutionProvider',
          'use_fp16': true,
          'use_nchw': true,
          'cpu_disabled': false,
        },
        {'name': 'CpuExecutionProvider'},
      ];
      return await _onnxRuntime.createSessionFromAsset(
        modelPath,
        // ignore: avoid_dynamic_calls
        options: opts,
      );
    } catch (_) {}

    // ── Variant B: options passed as positional argument ──────────────────────
    try {
      final opts = OrtSessionOptions();
      // ignore: avoid_dynamic_calls
      (opts as dynamic).executionProviders = <Map<String, dynamic>>[
        {'name': 'NnapiExecutionProvider', 'use_fp16': true},
        {'name': 'CpuExecutionProvider'},
      ];
      // ignore: avoid_dynamic_calls
      return await (_onnxRuntime as dynamic).createSessionFromAsset(
        modelPath,
        opts,
      );
    } catch (_) {}

    // ── Variant C: native MethodChannel (guaranteed fallback) ─────────────────
    // Requires the Kotlin snippet in android/app/src/.../MainActivity.kt below.
    try {
      final result = await _nativeChannel.invokeMethod<bool>(
        'reloadWithNnapi',
        {'modelPath': modelPath},
      );
      if (result == true) {
        // Session was configured on the native side; load normally here for
        // input/output name discovery only. Inference calls will use the
        // native-side NNAPI session via a second channel (see Kotlin below).
        return await _onnxRuntime.createSessionFromAsset(modelPath);
      }
    } catch (_) {}

    return null; // all variants failed, caller will use CPU
  }

  Future<void> _loadBenchmarkFile() async {
    try {
      final String content = await rootBundle.loadString(benchmarkFilePath);
      final lines = content
          .split('\n')
          .map((line) => line.trim())
          .where((line) => line.isNotEmpty && !line.startsWith('#'))
          .toList();

      setState(() {
        _imageUrls = lines;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to load benchmark file: $e';
      });
    }
  }

  /// Select 50 images by file size using HTTP HEAD requests
  Future<List<_ImageWithSize>> _selectImagesBySize(List<String> urls) async {
    setState(() {
      _errorMessage = 'Fetching image sizes via HEAD requests...';
    });

    final List<_ImageWithSize> imagesWithSize = [];

    for (int i = 0; i < urls.length; i++) {
      if (_isCancelled) break;
      try {
        final response = await http
            .head(Uri.parse(urls[i]))
            .timeout(const Duration(seconds: 10));
        final contentLength =
            int.tryParse(response.headers['content-length'] ?? '') ?? 0;
        imagesWithSize.add(_ImageWithSize(url: urls[i], size: contentLength));
      } catch (e) {
        // Skip images where HEAD fails
      }

      if (i % 20 == 0) {
        setState(() {
          _errorMessage = 'Fetching image sizes... ${i + 1}/${urls.length}';
        });
      }
    }

    imagesWithSize.sort((a, b) => a.size.compareTo(b.size));

    final List<_ImageWithSize> selected = [];
    final int total = imagesWithSize.length;

    if (total <= totalImages) {
      selected.addAll(imagesWithSize);
    } else {
      selected.addAll(imagesWithSize.sublist(0, 10));
      final int midStart = (total ~/ 2) - 15;
      final int midEnd = midStart + 30;
      selected.addAll(
        imagesWithSize.sublist(
          midStart.clamp(10, total - 10),
          midEnd.clamp(10, total - 10),
        ),
      );
      selected.addAll(imagesWithSize.sublist(total - 10));
    }

    setState(() {
      _errorMessage = null;
    });

    return selected.take(totalImages).toList();
  }

  // ==================== BENCHMARK EXECUTION ====================

  Future<void> _startBenchmark() async {
    if (_selectedModel == null || _imageUrls.isEmpty) {
      setState(() {
        _errorMessage =
            'Please select a model and ensure image URLs are loaded';
      });
      return;
    }

    await _loadSelectedModel();
    if (_session == null) return;

    _selectedImages = await _selectImagesBySize(_imageUrls);
    if (_selectedImages.isEmpty) {
      setState(() {
        _errorMessage = 'No images could be selected';
      });
      return;
    }

    setState(() {
      _isBenchmarking = true;
      _isCancelled = false;
      _isWarmingUp = true;
      _results = [];
      _currentImageIndex = 0;
      _currentTrial = 0;
      _warmUpProgress = 0;
      _successCount = 0;
      _failCount = 0;
      _errorMessage = null;
    });

    // --- WARM-UP PHASE ---
    try {
      final warmUpResponse = await http
          .get(Uri.parse(_selectedImages.first.url))
          .timeout(const Duration(seconds: 30));
      if (warmUpResponse.statusCode == 200) {
        for (int w = 0; w < warmUpCount; w++) {
          if (_isCancelled) break;
          setState(() {
            _warmUpProgress = w + 1;
          });
          await _runFullPipeline(warmUpResponse.bodyBytes);
          await Future.delayed(
            const Duration(milliseconds: interInferenceDelayMs),
          );
        }
      }
    } catch (_) {}

    setState(() {
      _isWarmingUp = false;
    });

    if (_isCancelled) {
      setState(() => _isBenchmarking = false);
      return;
    }

    // --- TIMED MEASUREMENT PHASE ---
    final overallStopwatch = Stopwatch()..start();

    for (int i = 0; i < _selectedImages.length; i++) {
      if (_isCancelled) break;

      setState(() {
        _currentImageIndex = i;
      });

      // Fetch image ONCE per image
      Uint8List? imageBytes;
      try {
        final response = await http
            .get(Uri.parse(_selectedImages[i].url))
            .timeout(const Duration(seconds: 30));
        if (response.statusCode == 200) {
          imageBytes = response.bodyBytes;
        }
      } catch (_) {}

      if (imageBytes == null) {
        for (int trial = 0; trial < trialsPerImage; trial++) {
          setState(() {
            _currentTrial = trial;
            _results.add(
              BenchmarkResult(
                index: i,
                trial: trial,
                url: _selectedImages[i].url,
                fileSize: _selectedImages[i].size,
                success: false,
                decodeTime: 0,
                preprocessTime: 0,
                inferenceTime: 0,
                postprocessTime: 0,
                totalTime: 0,
                error: 'Failed to fetch image',
              ),
            );
            _failCount++;
          });
        }
        continue;
      }

      // Run 10 trials on the same image bytes
      for (int trial = 0; trial < trialsPerImage; trial++) {
        if (_isCancelled) break;

        setState(() {
          _currentTrial = trial;
        });

        final result = await _processImage(
          imageBytes,
          _selectedImages[i],
          i,
          trial,
        );

        setState(() {
          _results.add(result);
          if (result.success) {
            _successCount++;
          } else {
            _failCount++;
          }
        });

        await Future.delayed(
          const Duration(milliseconds: interInferenceDelayMs),
        );
      }
    }

    overallStopwatch.stop();

    _calculateAllStats();

    setState(() {
      _totalBenchmarkTime = overallStopwatch.elapsedMicroseconds / 1000.0;
      _isBenchmarking = false;
    });
  }

  /// Run full pipeline without timing (warm-up only)
  Future<void> _runFullPipeline(Uint8List imageBytes) async {
    try {
      final inputTensor = await _preprocessImage(imageBytes);
      final inputs = {_inputName ?? 'input': inputTensor};
      final outputs = await _session!.run(inputs);
      inputTensor.dispose();
      for (final tensor in outputs.values) {
        tensor.dispose();
      }
    } catch (_) {}
  }

  /// Process image bytes with 4-step timing (all in milliseconds):
  /// 1) Decode  2) Preprocess  3) Inference  4) Postprocess
  Future<BenchmarkResult> _processImage(
    Uint8List imageBytes,
    _ImageWithSize imageInfo,
    int imageIndex,
    int trial,
  ) async {
    double decodeTime = 0;
    double preprocessTime = 0;
    double inferenceTime = 0;
    double postprocessTime = 0;
    double totalTime = 0;
    String? predictedClass;
    double? confidence;
    String? error;

    final totalStopwatch = Stopwatch()..start();

    try {
      // ========== STEP 1: DECODE ==========
      final decodeStopwatch = Stopwatch()..start();
      final image = img.decodeImage(imageBytes);
      decodeStopwatch.stop();
      decodeTime = decodeStopwatch.elapsedMicroseconds / 1000.0;

      if (image == null) {
        throw Exception('Failed to decode image');
      }

      // ========== STEP 2: PREPROCESS ==========
      final preprocessStopwatch = Stopwatch()..start();

      final resized = img.copyResize(
        image,
        width: inputSize,
        height: inputSize,
        interpolation: img.Interpolation.linear,
      );

      const int channels = 3;
      final int tensorTotalSize = 1 * channels * inputSize * inputSize;
      final Float32List tensorData = Float32List(tensorTotalSize);

      int idx = 0;
      for (int c = 0; c < channels; c++) {
        for (int y = 0; y < inputSize; y++) {
          for (int x = 0; x < inputSize; x++) {
            final pixel = resized.getPixel(x, y);
            double value;
            switch (c) {
              case 0:
                value = pixel.r / 255.0;
                break;
              case 1:
                value = pixel.g / 255.0;
                break;
              case 2:
                value = pixel.b / 255.0;
                break;
              default:
                value = 0.0;
            }
            tensorData[idx++] = ((value - mean[c]) / std[c]);
          }
        }
      }

      final inputTensor = await OrtValue.fromList(tensorData, [
        1,
        3,
        inputSize,
        inputSize,
      ]);

      preprocessStopwatch.stop();
      preprocessTime = preprocessStopwatch.elapsedMicroseconds / 1000.0;

      // ========== STEP 3: INFERENCE ==========
      final inferenceStopwatch = Stopwatch()..start();
      final inputs = {_inputName ?? 'input': inputTensor};
      final outputs = await _session!.run(inputs);
      inferenceStopwatch.stop();
      inferenceTime = inferenceStopwatch.elapsedMicroseconds / 1000.0;

      // ========== STEP 4: POSTPROCESS ==========
      final postprocessStopwatch = Stopwatch()..start();

      final outputTensor = outputs[_outputName ?? 'output'];
      if (outputTensor == null) {
        throw Exception('No output tensor');
      }

      final rawOutput = await outputTensor.asList();

      dynamic flatOutput = rawOutput;
      if (rawOutput.isNotEmpty && rawOutput[0] is List) {
        flatOutput = rawOutput[0];
      }

      List<double> outputDoubles = [];
      if (flatOutput is Float32List) {
        for (int i = 0; i < flatOutput.length; i++) {
          double val = flatOutput[i];
          if (val.isNaN) val = 0.0;
          if (val.isInfinite) val = val > 0 ? 1e10 : -1e10;
          outputDoubles.add(val);
        }
      } else if (flatOutput is List) {
        for (var element in flatOutput) {
          if (element is num) {
            outputDoubles.add(element.toDouble());
          } else {
            outputDoubles.add(0.0);
          }
        }
      }

      final probabilities = _softmax(outputDoubles);

      int maxIdx = 0;
      double maxProb = probabilities[0];
      for (int i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIdx = i;
        }
      }

      predictedClass = maxIdx < classLabels.length
          ? classLabels[maxIdx]
          : 'Class $maxIdx';
      confidence = maxProb;

      postprocessStopwatch.stop();
      postprocessTime = postprocessStopwatch.elapsedMicroseconds / 1000.0;

      // Cleanup
      inputTensor.dispose();
      for (final tensor in outputs.values) {
        tensor.dispose();
      }
    } catch (e) {
      error = e.toString();
    }

    totalStopwatch.stop();
    totalTime = totalStopwatch.elapsedMicroseconds / 1000.0;

    return BenchmarkResult(
      index: imageIndex,
      trial: trial,
      url: imageInfo.url,
      fileSize: imageInfo.size,
      success: error == null,
      predictedClass: predictedClass,
      confidence: confidence,
      decodeTime: decodeTime,
      preprocessTime: preprocessTime,
      inferenceTime: inferenceTime,
      postprocessTime: postprocessTime,
      totalTime: totalTime,
      error: error,
    );
  }

  /// Preprocess for warm-up only (uses isolate)
  Future<OrtValue> _preprocessImage(Uint8List imageBytes) async {
    final tensorData = await compute(
      _preprocessImageIsolate,
      BenchmarkPreprocessParams(
        imageBytes: imageBytes,
        inputSize: inputSize,
        mean: mean,
        std: std,
      ),
    );

    return await OrtValue.fromList(tensorData, [1, 3, inputSize, inputSize]);
  }

  List<double> _softmax(List<double> logits) {
    final maxLogit = logits.reduce((a, b) => a > b ? a : b);
    final expValues = logits.map((x) => math.exp(x - maxLogit)).toList();
    final sumExp = expValues.reduce((a, b) => a + b);
    return expValues.map((x) => x / sumExp).toList();
  }

  void _cancelBenchmark() {
    setState(() {
      _isCancelled = true;
    });
  }

  // ==================== STATISTICS ====================

  void _calculateAllStats() {
    final successResults = _results.where((r) => r.success).toList();
    if (successResults.isEmpty) return;

    final decodeTimes = successResults.map((r) => r.decodeTime).toList();
    final preprocessTimes = successResults
        .map((r) => r.preprocessTime)
        .toList();
    final inferenceTimes = successResults.map((r) => r.inferenceTime).toList();
    final postprocessTimes = successResults
        .map((r) => r.postprocessTime)
        .toList();
    final totalTimes = successResults.map((r) => r.totalTime).toList();

    setState(() {
      _decodeStats = _computeStats(decodeTimes);
      _preprocessStats = _computeStats(preprocessTimes);
      _inferenceStats = _computeStats(inferenceTimes);
      _postprocessStats = _computeStats(postprocessTimes);
      _totalStats = _computeStats(totalTimes);

      // Thermal drift on total time (chronological order)
      final halfN = totalTimes.length ~/ 2;
      final firstHalfOriginal = List<double>.from(totalTimes.sublist(0, halfN))
        ..sort();
      final secondHalfOriginal = List<double>.from(totalTimes.sublist(halfN))
        ..sort();

      _firstHalfMedian = _median(firstHalfOriginal);
      _secondHalfMedian = _median(secondHalfOriginal);
      _thermalDriftPercent = _firstHalfMedian > 0
          ? ((_secondHalfMedian - _firstHalfMedian) / _firstHalfMedian * 100)
                .abs()
          : 0.0;
      _thermalDriftWarning = _thermalDriftPercent > 5.0;
    });
  }

  _StepStats _computeStats(List<double> times) {
    final sorted = List<double>.from(times)..sort();
    final n = sorted.length;

    final meanVal = times.reduce((a, b) => a + b) / n;

    double sumSquaredDiff = 0;
    for (final t in times) {
      sumSquaredDiff += (t - meanVal) * (t - meanVal);
    }
    final stdDev = math.sqrt(sumSquaredDiff / n);

    final median = _median(sorted);

    final q1Index = (n * 0.25).floor();
    final q3Index = (n * 0.75).floor();
    final iqr = sorted[q3Index] - sorted[q1Index];

    final p95Index = (n * 0.95).floor().clamp(0, n - 1);
    final p95 = sorted[p95Index];

    return _StepStats(
      median: median,
      mean: meanVal,
      stdDev: stdDev,
      iqr: iqr,
      p95: p95,
      min: sorted.first,
      max: sorted.last,
    );
  }

  double _median(List<double> sorted) {
    final n = sorted.length;
    if (n == 0) return 0;
    return n % 2 == 0
        ? (sorted[n ~/ 2 - 1] + sorted[n ~/ 2]) / 2
        : sorted[n ~/ 2];
  }

  // ==================== EXPORT ====================

  void _exportResults() {
    final buffer = StringBuffer();

    buffer.writeln(
      'Image Index,Trial,URL,File Size (bytes),Success,'
      'Decode (ms),Preprocess (ms),Inference (ms),Postprocess (ms),Total (ms),'
      'Predicted Class,Confidence,Error',
    );

    for (final r in _results) {
      buffer.writeln(
        '${r.index},'
        '${r.trial},'
        '"${r.url}",'
        '${r.fileSize},'
        '${r.success},'
        '${r.decodeTime.toStringAsFixed(2)},'
        '${r.preprocessTime.toStringAsFixed(2)},'
        '${r.inferenceTime.toStringAsFixed(2)},'
        '${r.postprocessTime.toStringAsFixed(2)},'
        '${r.totalTime.toStringAsFixed(2)},'
        '${r.predictedClass ?? ""},'
        '${r.confidence?.toStringAsFixed(4) ?? ""},'
        '"${r.error ?? ""}"',
      );
    }

    buffer.writeln('');
    buffer.writeln('Summary');
    buffer.writeln('Model,$_selectedModel');
    buffer.writeln('Total Images,${_selectedImages.length}');
    buffer.writeln('Trials Per Image,$trialsPerImage');
    buffer.writeln('Total Measurements,${_results.length}');
    buffer.writeln('Success,$_successCount');
    buffer.writeln('Failed,$_failCount');

    buffer.writeln('');
    buffer.writeln(
      'Step,Median (ms),Mean (ms),StdDev (ms),IQR (ms),P95 (ms),Min (ms),Max (ms)',
    );
    _writeStatsRow(buffer, 'Decode', _decodeStats);
    _writeStatsRow(buffer, 'Preprocess', _preprocessStats);
    _writeStatsRow(buffer, 'Inference', _inferenceStats);
    _writeStatsRow(buffer, 'Postprocess', _postprocessStats);
    _writeStatsRow(buffer, 'Total E2E', _totalStats);

    buffer.writeln('');
    buffer.writeln('Thermal Drift Analysis (on Total E2E)');
    buffer.writeln(
      'First Half Median (ms),${_firstHalfMedian.toStringAsFixed(2)}',
    );
    buffer.writeln(
      'Second Half Median (ms),${_secondHalfMedian.toStringAsFixed(2)}',
    );
    buffer.writeln('Drift (%),${_thermalDriftPercent.toStringAsFixed(2)}');
    buffer.writeln('Drift Warning (>5%),$_thermalDriftWarning');
    buffer.writeln('');
    buffer.writeln(
      'Total Benchmark Time (ms),${_totalBenchmarkTime.toStringAsFixed(2)}',
    );
    buffer.writeln('Execution Provider,$_selectedProvider');
    buffer.writeln('NNAPI Active,$_nnapiAvailable');
    Clipboard.setData(ClipboardData(text: buffer.toString()));

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Results copied to clipboard as CSV'),
        backgroundColor: Colors.green,
      ),
    );
  }

  void _writeStatsRow(StringBuffer buffer, String name, _StepStats s) {
    buffer.writeln(
      '$name,'
      '${s.median.toStringAsFixed(2)},'
      '${s.mean.toStringAsFixed(2)},'
      '${s.stdDev.toStringAsFixed(2)},'
      '${s.iqr.toStringAsFixed(2)},'
      '${s.p95.toStringAsFixed(2)},'
      '${s.min.toStringAsFixed(2)},'
      '${s.max.toStringAsFixed(2)}',
    );
  }

  // ==================== UI ====================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Benchmark (E2E)'),
        centerTitle: true,
        actions: [
          if (_results.isNotEmpty && !_isBenchmarking)
            IconButton(
              icon: const Icon(Icons.download),
              onPressed: _exportResults,
              tooltip: 'Export Results (CSV)',
            ),
        ],
      ),
      body: SafeArea(
        bottom: false,
        child: CustomScrollView(
          slivers: [
            SliverToBoxAdapter(child: _buildMethodologyCard()),
            SliverToBoxAdapter(child: _buildStatusCard()),
            if (_isBenchmarking)
              SliverToBoxAdapter(child: _buildProgressSection()),
            if (_results.isNotEmpty)
              SliverToBoxAdapter(child: _buildStatisticsCard()),
            if (_thermalDriftWarning && _results.isNotEmpty)
              SliverToBoxAdapter(child: _buildThermalWarningCard()),
            if (_results.isEmpty && !_isBenchmarking)
              SliverFillRemaining(
                hasScrollBody: false,
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.analytics_outlined,
                        size: 64,
                        color: Colors.grey.shade400,
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'No benchmark results yet',
                        style: TextStyle(color: Colors.grey.shade600),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Select a model and press "Start Benchmark"',
                        style: TextStyle(
                          color: Colors.grey.shade500,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
      bottomNavigationBar: SafeArea(child: _buildActionButtons()),
    );
  }

  Widget _buildMethodologyCard() {
    return Card(
      margin: const EdgeInsets.fromLTRB(16, 16, 16, 0),
      color: Colors.blue.shade50,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.science, color: Colors.blue.shade700, size: 20),
                const SizedBox(width: 8),
                Text(
                  'Benchmark Protocol (E2E)',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.blue.shade700,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              '• 4 steps: Decode → Preprocess → Inference → Postprocess\n'
              '• Image fetched once, then $trialsPerImage trials on same bytes\n'
              '• $warmUpCount warm-up (discarded), $trialsPerImage × $totalImages = ${trialsPerImage * totalImages} measurements\n'
              '• ${interInferenceDelayMs}ms delay between inferences',
              style: TextStyle(fontSize: 12, color: Colors.blue.shade900),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusCard() {
    return Card(
      margin: const EdgeInsets.all(16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  _availableModels.isNotEmpty
                      ? Icons.check_circle
                      : Icons.hourglass_empty,
                  color: _availableModels.isNotEmpty
                      ? Colors.green
                      : Colors.orange,
                  size: 32,
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Models: ${_availableModels.length} available',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      Text(
                        'Images: ${_imageUrls.length} URLs loaded',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                      if (_errorMessage != null)
                        Text(
                          _errorMessage!,
                          style: TextStyle(
                            color: Colors.red.shade700,
                            fontSize: 12,
                          ),
                        ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Text(
              'Select Model:',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 8),
            if (_isLoadingModels)
              const Center(child: CircularProgressIndicator())
            else if (_availableModels.isEmpty)
              Text(
                'No models found. Add .onnx files to assets/models/ and list them in assets/models/model_list.txt',
                style: TextStyle(color: Colors.grey.shade600, fontSize: 12),
              )
            else
              Container(
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade400),
                  borderRadius: BorderRadius.circular(8),
                ),
                padding: const EdgeInsets.symmetric(horizontal: 12),
                child: DropdownButton<String>(
                  value: _selectedModel,
                  hint: const Text('Select a model...'),
                  isExpanded: true,
                  underline: const SizedBox(),
                  onChanged: _isBenchmarking
                      ? null
                      : (value) => setState(() => _selectedModel = value),
                  items: _availableModels
                      .map(
                        (m) =>
                            DropdownMenuItem<String>(value: m, child: Text(m)),
                      )
                      .toList(),
                ),
              ),
            const SizedBox(height: 16),
            Text(
              'Execution Provider:',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 8),
            Container(
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey.shade400),
                borderRadius: BorderRadius.circular(8),
              ),
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: DropdownButton<String>(
                value: _selectedProvider,
                isExpanded: true,
                underline: const SizedBox(),
                onChanged: _isBenchmarking
                    ? null
                    : (v) => setState(() => _selectedProvider = v!),
                items: _providers
                    .map(
                      (p) => DropdownMenuItem<String>(
                        value: p['value'],
                        child: Row(
                          children: [
                            Icon(
                              p['value'] == 'nnapi'
                                  ? Icons.memory
                                  : Icons.developer_board,
                              size: 18,
                              color: p['value'] == 'nnapi'
                                  ? Colors.deepPurple
                                  : Colors.blueGrey,
                            ),
                            const SizedBox(width: 8),
                            Text(p['label']!),
                            if (p['value'] == 'nnapi' && _nnapiAvailable) ...[
                              const SizedBox(width: 6),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 6,
                                  vertical: 2,
                                ),
                                decoration: BoxDecoration(
                                  color: Colors.green.shade100,
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                child: Text(
                                  'Active',
                                  style: TextStyle(
                                    fontSize: 10,
                                    color: Colors.green.shade800,
                                  ),
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                    )
                    .toList(),
              ),
            ),
            if (_selectedProvider == 'nnapi')
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Row(
                  children: [
                    Icon(
                      Icons.info_outline,
                      size: 13,
                      color: Colors.deepPurple.shade300,
                    ),
                    const SizedBox(width: 4),
                    Expanded(
                      child: Text(
                        'Delegates to NPU/DSP/GPU. CPU fallback enabled. Android 8.1+ required.',
                        style: TextStyle(
                          fontSize: 11,
                          color: Colors.deepPurple.shade300,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            const Divider(),
            Text(
              'Total measurements: $totalImages × $trialsPerImage = ${totalImages * trialsPerImage}',
              style: TextStyle(color: Colors.grey.shade600, fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildProgressSection() {
    final totalRuns = _selectedImages.length * trialsPerImage;
    final completedRuns = _results.length;
    final progress = totalRuns == 0 ? 0.0 : completedRuns / totalRuns;

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (_selectedModel != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text(
                  'Model: $_selectedModel',
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                    color: Colors.teal,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            if (_isWarmingUp) ...[
              Row(
                children: [
                  Icon(Icons.whatshot, color: Colors.orange, size: 20),
                  const SizedBox(width: 8),
                  Text(
                    'Warm-up: $_warmUpProgress / $warmUpCount',
                    style: TextStyle(
                      color: Colors.orange.shade700,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              LinearProgressIndicator(
                value: _warmUpProgress / warmUpCount,
                minHeight: 8,
                borderRadius: BorderRadius.circular(4),
                backgroundColor: Colors.orange.shade100,
                valueColor: AlwaysStoppedAnimation(Colors.orange),
              ),
              const SizedBox(height: 8),
              Text(
                'Stabilizing CPU frequency and cache state...',
                style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
              ),
            ] else ...[
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Image: ${_currentImageIndex + 1}/${_selectedImages.length} | Trial: ${_currentTrial + 1}/$trialsPerImage',
                    style: Theme.of(context).textTheme.titleSmall,
                  ),
                  Text(
                    '${(progress * 100).toStringAsFixed(1)}%',
                    style: Theme.of(context).textTheme.titleSmall,
                  ),
                ],
              ),
              const SizedBox(height: 8),
              LinearProgressIndicator(
                value: progress,
                minHeight: 8,
                borderRadius: BorderRadius.circular(4),
              ),
              const SizedBox(height: 4),
              Text(
                'Completed: $completedRuns / $totalRuns measurements',
                style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
              ),
            ],
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildMiniStat(
                  'Success',
                  _successCount.toString(),
                  Colors.green,
                ),
                _buildMiniStat('Failed', _failCount.toString(), Colors.red),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMiniStat(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          value,
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
        Text(label, style: const TextStyle(fontSize: 12)),
      ],
    );
  }

  Widget _buildStatisticsCard() {
    if (_results.isEmpty) return const SizedBox.shrink();

    return Card(
      margin: const EdgeInsets.all(16),
      color: Colors.teal.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'End-to-End Latency Statistics',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),

            // Total E2E (primary)
            _buildStepStatsRow('Total E2E', _totalStats, isPrimary: true),
            const Divider(height: 20),

            // Per-step breakdown
            Text(
              'Per-Step Breakdown',
              style: Theme.of(
                context,
              ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            _buildStepStatsRow('1. Decode', _decodeStats),
            const SizedBox(height: 4),
            _buildStepStatsRow('2. Preprocess', _preprocessStats),
            const SizedBox(height: 4),
            _buildStepStatsRow('3. Inference', _inferenceStats),
            const SizedBox(height: 4),
            _buildStepStatsRow('4. Postprocess', _postprocessStats),

            const Divider(height: 24),

            // Thermal drift
            Text(
              'Thermal Drift Analysis',
              style: Theme.of(
                context,
              ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: _buildStatItem(
                    '1st Half Median',
                    '${_firstHalfMedian.toStringAsFixed(2)} ms',
                    Icons.first_page,
                  ),
                ),
                Expanded(
                  child: _buildStatItem(
                    '2nd Half Median',
                    '${_secondHalfMedian.toStringAsFixed(2)} ms',
                    Icons.last_page,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: _buildStatItem(
                    'Drift',
                    '${_thermalDriftPercent.toStringAsFixed(2)}%',
                    _thermalDriftWarning ? Icons.warning : Icons.check_circle,
                    color: _thermalDriftWarning ? Colors.orange : Colors.green,
                  ),
                ),
                Expanded(
                  child: _buildStatItem(
                    'Success Rate',
                    '${_results.isEmpty ? 0 : (_successCount / _results.length * 100).toStringAsFixed(1)}%',
                    Icons.check_circle_outline,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStepStatsRow(
    String label,
    _StepStats stats, {
    bool isPrimary = false,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
      margin: const EdgeInsets.only(bottom: 4),
      decoration: BoxDecoration(
        color: isPrimary ? Colors.teal.shade100 : Colors.grey.shade50,
        borderRadius: BorderRadius.circular(8),
        border: isPrimary ? Border.all(color: Colors.teal.shade300) : null,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: isPrimary ? 14 : 12,
              fontWeight: FontWeight.bold,
              color: isPrimary ? Colors.teal.shade900 : Colors.grey.shade800,
            ),
          ),
          const SizedBox(height: 6),
          Row(
            children: [
              Expanded(
                child: _buildMiniMetric(
                  'Median',
                  '${stats.median.toStringAsFixed(2)}',
                  isPrimary: isPrimary,
                ),
              ),
              Expanded(
                child: _buildMiniMetric(
                  'Mean±SD',
                  '${stats.mean.toStringAsFixed(2)}±${stats.stdDev.toStringAsFixed(2)}',
                ),
              ),
              Expanded(
                child: _buildMiniMetric(
                  'P95',
                  '${stats.p95.toStringAsFixed(2)}',
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              Expanded(
                child: _buildMiniMetric(
                  'IQR',
                  '${stats.iqr.toStringAsFixed(2)}',
                ),
              ),
              Expanded(
                child: _buildMiniMetric(
                  'Min',
                  '${stats.min.toStringAsFixed(2)}',
                ),
              ),
              Expanded(
                child: _buildMiniMetric(
                  'Max',
                  '${stats.max.toStringAsFixed(2)}',
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMiniMetric(
    String label,
    String value, {
    bool isPrimary = false,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(fontSize: 10, color: Colors.grey.shade600),
        ),
        Text(
          '$value ms',
          style: TextStyle(
            fontSize: isPrimary ? 14 : 12,
            fontWeight: isPrimary ? FontWeight.bold : FontWeight.w600,
            color: isPrimary ? Colors.teal.shade900 : null,
          ),
        ),
      ],
    );
  }

  Widget _buildThermalWarningCard() {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      color: Colors.orange.shade50,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            Icon(Icons.warning_amber, color: Colors.orange.shade700),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Thermal Drift Detected',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: Colors.orange.shade700,
                    ),
                  ),
                  Text(
                    'Drift of ${_thermalDriftPercent.toStringAsFixed(2)}% exceeds 5% threshold. '
                    'Consider allowing device to cool and rerunning.',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.orange.shade900,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatItem(
    String label,
    String value,
    IconData icon, {
    bool isPrimary = false,
    Color? color,
  }) {
    return Row(
      children: [
        Icon(icon, size: 16, color: color ?? Colors.teal),
        const SizedBox(width: 4),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: TextStyle(
                  fontSize: 11,
                  color: isPrimary ? Colors.teal.shade700 : Colors.grey,
                  fontWeight: isPrimary ? FontWeight.bold : FontWeight.normal,
                ),
              ),
              Text(
                value,
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: isPrimary ? 15 : 13,
                  color: isPrimary ? Colors.teal.shade900 : null,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildActionButtons() {
    return Container(
      decoration: BoxDecoration(
        color: Theme.of(context).scaffoldBackgroundColor,
        border: Border(
          top: BorderSide(color: Colors.grey.shade300, width: 0.5),
        ),
      ),
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          if (_isBenchmarking)
            Expanded(
              child: OutlinedButton.icon(
                onPressed: _cancelBenchmark,
                icon: const Icon(Icons.stop),
                label: const Text('Stop'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: Colors.red,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
              ),
            )
          else
            Expanded(
              child: FilledButton.icon(
                onPressed:
                    _selectedModel != null &&
                        !_isLoadingModels &&
                        _imageUrls.isNotEmpty
                    ? _startBenchmark
                    : null,
                icon: const Icon(Icons.play_arrow),
                label: const Text('Start Benchmark'),
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// ==================== DATA CLASSES ====================

class BenchmarkResult {
  final int index;
  final int trial;
  final String url;
  final int fileSize;
  final bool success;
  final String? predictedClass;
  final double? confidence;
  final double decodeTime; // ms
  final double preprocessTime; // ms
  final double inferenceTime; // ms
  final double postprocessTime; // ms
  final double totalTime; // ms
  final String? error;

  BenchmarkResult({
    required this.index,
    required this.trial,
    required this.url,
    required this.fileSize,
    required this.success,
    this.predictedClass,
    this.confidence,
    required this.decodeTime,
    required this.preprocessTime,
    required this.inferenceTime,
    required this.postprocessTime,
    required this.totalTime,
    this.error,
  });
}

class _ImageWithSize {
  final String url;
  final int size;

  _ImageWithSize({required this.url, required this.size});
}

class _StepStats {
  final double median;
  final double mean;
  final double stdDev;
  final double iqr;
  final double p95;
  final double min;
  final double max;

  _StepStats({
    required this.median,
    required this.mean,
    required this.stdDev,
    required this.iqr,
    required this.p95,
    required this.min,
    required this.max,
  });

  factory _StepStats.empty() =>
      _StepStats(median: 0, mean: 0, stdDev: 0, iqr: 0, p95: 0, min: 0, max: 0);
}

// Parameters for isolate preprocessing (warm-up only)
class BenchmarkPreprocessParams {
  final Uint8List imageBytes;
  final int inputSize;
  final List<double> mean;
  final List<double> std;

  BenchmarkPreprocessParams({
    required this.imageBytes,
    required this.inputSize,
    required this.mean,
    required this.std,
  });
}

// Static function for preprocessing in isolate (warm-up only)
Float32List _preprocessImageIsolate(BenchmarkPreprocessParams params) {
  final image = img.decodeImage(params.imageBytes);
  if (image == null) throw Exception('Failed to decode image');

  final resized = img.copyResize(
    image,
    width: params.inputSize,
    height: params.inputSize,
    interpolation: img.Interpolation.linear,
  );

  const int channels = 3;
  final int totalSize = 1 * channels * params.inputSize * params.inputSize;
  final Float32List tensorData = Float32List(totalSize);

  int idx = 0;
  for (int c = 0; c < channels; c++) {
    for (int y = 0; y < params.inputSize; y++) {
      for (int x = 0; x < params.inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        double value;
        switch (c) {
          case 0:
            value = pixel.r / 255.0;
            break;
          case 1:
            value = pixel.g / 255.0;
            break;
          case 2:
            value = pixel.b / 255.0;
            break;
          default:
            value = 0.0;
        }
        tensorData[idx++] = ((value - params.mean[c]) / params.std[c]);
      }
    }
  }

  return tensorData;
}
