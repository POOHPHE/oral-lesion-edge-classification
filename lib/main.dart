import 'package:flutter/material.dart';
import 'benchmark_page.dart';
import 'classify_page.dart'; // Import the new page

void main() {
  runApp(const OralLesionClassifierApp());
}

class OralLesionClassifierApp extends StatelessWidget {
  const OralLesionClassifierApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Oral Lesion Classifier',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.teal,
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _currentIndex = 0;

  // Add ClassifyPage to the list
  final List<Widget> _pages = const [
    ClassifyPage(),
    BenchmarkPage(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack( // Use IndexedStack to preserve page state
        index: _currentIndex,
        children: _pages,
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.image_search),
            label: 'Classify',
          ),
          NavigationDestination(
            icon: Icon(Icons.speed_outlined),
            label: 'Benchmark',
          ),
        ],
      ),
    );
  }
}