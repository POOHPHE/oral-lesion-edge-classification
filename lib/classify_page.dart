import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

class ClassifyPage extends StatefulWidget {
  const ClassifyPage({super.key});

  @override
  State<ClassifyPage> createState() => _ClassifyPageState();
}

class _ClassifyPageState extends State<ClassifyPage> {
  final OnnxRuntime _onnxRuntime = OnnxRuntime();
  OrtSession? _session;
  final ImagePicker _picker = ImagePicker();

  File? _image;
  bool _isProcessing = false;
  bool _isModelLoading = true;
  String _resultLabel = "Select an image to start";
  String _debugError = "";
  double _confidence = 0.0;

  // --- UPDATED: ConvNeXt Pico Model ---
  static const String modelPath = 'assets/models/convnext_pico_int8.onnx';
  static const int inputSize = 224;

  static const List<String> classLabels = [
    'Mass',
    'Normal',
    'Pigmented',
    'Red',
    'Ulcer and bullous',
    'White and mixed white-red',
  ];

  @override
  void initState() {
    super.initState();
    _initializeModel();
  }

  Future<void> _initializeModel() async {
    try {
      _session = await _onnxRuntime.createSessionFromAsset(modelPath);
      debugPrint("Model Inputs: ${_session?.inputNames}");
      debugPrint("Model Outputs: ${_session?.outputNames}");
      setState(() => _isModelLoading = false);
    } catch (e) {
      setState(() {
        _isModelLoading = false;
        _resultLabel = "Model failed to load";
        _debugError = e.toString();
      });
    }
  }

  Future<void> _pickAndClassify(ImageSource source) async {
    if (_session == null) return;

    final XFile? pickedFile = await _picker.pickImage(source: source);
    if (pickedFile == null) return;

    setState(() {
      _image = File(pickedFile.path);
      _isProcessing = true;
      _debugError = "";
    });

    try {
      final bytes = await pickedFile.readAsBytes();
      final result = await _runInference(bytes);

      setState(() {
        _resultLabel = result['label'];
        _confidence = result['confidence'];
        _isProcessing = false;
      });
    } catch (e) {
      debugPrint("Inference Exception: $e");
      setState(() {
        _resultLabel = "Error during inference";
        _debugError = e
            .toString()
            .split('\n')
            .first; // Show first line of error
        _isProcessing = false;
      });
    }
  }

  Future<Map<String, dynamic>> _runInference(Uint8List bytes) async {
    final image = img.decodeImage(bytes);
    if (image == null) throw Exception("Decode failed");

    final resized = img.copyResize(image, width: inputSize, height: inputSize);

    // --- Float32List and ImageNet Normalization ---
    final Float32List tensorData = Float32List(1 * 3 * inputSize * inputSize);

    final mean = [0.485, 0.456, 0.406];
    final std = [0.229, 0.224, 0.225];

    int idx = 0;
    // Planar format (CHW)
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < inputSize; y++) {
        for (int x = 0; x < inputSize; x++) {
          final pixel = resized.getPixel(x, y);

          int val = (c == 0 ? pixel.r : (c == 1 ? pixel.g : pixel.b)).toInt();

          tensorData[idx++] = ((val / 255.0) - mean[c]) / std[c];
        }
      }
    }

    final inputName = _session!.inputNames.first;
    final inputTensor = await OrtValue.fromList(tensorData, [
      1,
      3,
      inputSize,
      inputSize,
    ]);

    final outputs = await _session!.run({inputName: inputTensor});

    final outputTensor = outputs.values.first;
    final List<dynamic> rawOutput = await outputTensor.asList();

    // Flatten output in case it is nested [[...]]
    List<double> logits = [];
    if (rawOutput[0] is List) {
      logits = (rawOutput[0] as List)
          .map((e) => (e as num).toDouble())
          .toList();
    } else {
      logits = rawOutput.map((e) => (e as num).toDouble()).toList();
    }

    // Diagnostic print to verify output size (should be 6, not 1000)
    debugPrint("Logits length: ${logits.length}");

    // Softmax
    final maxLogit = logits.reduce(math.max);
    final exps = logits.map((x) => math.exp(x - maxLogit)).toList();
    final sumExps = exps.reduce((a, b) => a + b);
    final probs = exps.map((x) => x / sumExps).toList();

    int maxIdx = 0;
    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }

    inputTensor.dispose();
    for (var o in outputs.values) {
      o.dispose();
    }

    return {
      'label': maxIdx < classLabels.length ? classLabels[maxIdx] : "Unknown",
      'confidence': probs[maxIdx],
    };
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Oral Lesion Classification"),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Column(
        children: [
          Expanded(
            child: Container(
              margin: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.grey.shade300),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: _image == null
                    ? Center(
                        child: Icon(
                          Icons.image,
                          size: 80,
                          color: Colors.grey.shade400,
                        ),
                      )
                    : Image.file(_image!, fit: BoxFit.contain),
              ),
            ),
          ),
          if (_isProcessing || _isModelLoading) const LinearProgressIndicator(),
          _buildResultCard(),
        ],
      ),
    );
  }

  Widget _buildResultCard() {
    return Card(
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 24),
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              _resultLabel,
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: _debugError.isNotEmpty
                    ? Colors.red
                    : Colors.teal.shade800,
              ),
            ),
            if (_debugError.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  "Debug: $_debugError",
                  style: const TextStyle(
                    fontSize: 11,
                    color: Colors.red,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ),
            if (!_isModelLoading && _confidence > 0 && _debugError.isEmpty)
              Text(
                "Confidence: ${(_confidence * 100).toStringAsFixed(1)}%",
                style: TextStyle(fontSize: 16, color: Colors.grey.shade700),
              ),
            const SizedBox(height: 24),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _isModelLoading || _isProcessing
                        ? null
                        : () => _pickAndClassify(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text("Camera"),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _isModelLoading || _isProcessing
                        ? null
                        : () => _pickAndClassify(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text("Gallery"),
                    style: FilledButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _session?.close();
    super.dispose();
  }
}
