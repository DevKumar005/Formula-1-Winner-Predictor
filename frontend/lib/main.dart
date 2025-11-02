import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(F1PredictorApp());
}

class F1PredictorApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'F1 Winner Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.red,
        scaffoldBackgroundColor: Color(0xFF181818),
        fontFamily: 'Roboto',
      ),
      home: AppNavigator(),
    );
  }
}

class AppNavigator extends StatefulWidget {
  @override
  _AppNavigatorState createState() => _AppNavigatorState();
}

class _AppNavigatorState extends State<AppNavigator> {
  int _selectedIndex = 0;

  final List<Widget> _screens = <Widget>[
    HomeScreen(),
    PredictionsScreen(),
    DriversScreen(),
    RaceInfoScreen(),
  ];

  static const List<String> _titles = [
    "Home",
    "Predictions",
    "Drivers",
    "Race Info"
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_titles[_selectedIndex]),
        centerTitle: true,
      ),
      body: _screens[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        selectedItemColor: Colors.red[700],
        unselectedItemColor: Colors.grey[400],
        backgroundColor: Color(0xFF191919),
        showUnselectedLabels: true,
        type: BottomNavigationBarType.fixed,
        onTap: _onItemTapped,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: "Home",
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.bar_chart),
            label: "Predictions",
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.people),
            label: "Drivers",
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.info),
            label: "Race Info",
          ),
        ],
      ),
    );
  }
}

// --- Home Screen ---
class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: EdgeInsets.all(24),
        child: Text(
          "Welcome to the F1 Winner Predictor!\nGet data-driven predictions for the Las Vegas Grand Prix 2025.",
          style: TextStyle(fontSize: 22, color: Colors.white70),
          textAlign: TextAlign.center,
        ),
      ),
    );
  }
}

// --- Predictions Screen ---
class PredictionsScreen extends StatefulWidget {
  @override
  _PredictionsScreenState createState() => _PredictionsScreenState();
}

class _PredictionsScreenState extends State<PredictionsScreen> {
  List<dynamic> _predictions = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    fetchPredictions();
  }

  Future<void> fetchPredictions() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final response = await http.get(Uri.parse('http://127.0.0.1:5000/api/predictions'));
      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        setState(() {
          _predictions = data;
          _loading = false;
        });
      } else {
        setState(() {
          _error = 'Server error: ${response.statusCode}';
          _loading = false;
        });
      }
    } catch (e) {
      setState(() {
        _error = 'Error fetching data: $e';
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(child: Text(_error!, style: TextStyle(color: Colors.red)));
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Las Vegas Grand Prix Winner Predictions',
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.bold,
              color: Colors.red[600],
            ),
          ),
          SizedBox(height: 16),
          Expanded(
            child: ListView.builder(
              itemCount: _predictions.length,
              itemBuilder: (context, index) {
                final driver = _predictions[index];
                return Card(
                  color: Color(0xFF222222),
                  elevation: 5,
                  margin: EdgeInsets.symmetric(vertical: 8, horizontal: 6),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  child: ListTile(
                    leading: CircleAvatar(
                      backgroundColor: Colors.red[800],
                      child: Text(
                        '${index + 1}',
                        style: TextStyle(color: Colors.white),
                      ),
                    ),
                    title: Text(
                      driver['name'],
                      style: TextStyle(fontSize: 18, color: Colors.white),
                    ),
                    trailing: Text(
                      '${(driver['probability'] * 100).toStringAsFixed(1)}%',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Colors.red[400],
                        fontSize: 18,
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

// --- Drivers Screen ---
class DriversScreen extends StatefulWidget {
  @override
  _DriversScreenState createState() => _DriversScreenState();
}

class _DriversScreenState extends State<DriversScreen> {
  List<dynamic> _drivers = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    fetchDrivers();
  }

  Future<void> fetchDrivers() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final response = await http.get(Uri.parse('http://127.0.0.1:5000/api/drivers'));
      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        setState(() {
          _drivers = data;
          _loading = false;
        });
      } else {
        setState(() {
          _error = 'Server error: ${response.statusCode}';
          _loading = false;
        });
      }
    } catch (e) {
      setState(() {
        _error = 'Error fetching data: $e';
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(child: Text(_error!, style: TextStyle(color: Colors.red)));
    }

    return ListView.builder(
      padding: EdgeInsets.all(12),
      itemCount: _drivers.length,
      itemBuilder: (context, index) {
        final driver = _drivers[index];
        return Card(
          color: Color(0xFF222222),
          margin: EdgeInsets.symmetric(vertical: 8),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          child: ListTile(
            leading: CircleAvatar(
              radius: 30,
              backgroundColor: Colors.transparent,
              backgroundImage: NetworkImage(driver['profile_img']),
            ),
            title: Text(
              driver['name'],
              style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 18),
            ),
            subtitle: Text(
              driver['team'],
              style: TextStyle(color: Colors.grey[300]),
            ),
            trailing: Container(
              width: 100,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text("Points: ${driver['points']}", style: TextStyle(color: Colors.white)),
                  SizedBox(height: 2),
                  Text("Podiums: ${driver['podiums']}", style: TextStyle(color: Colors.white)),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

// --- Race Info Screen ---
class RaceInfoScreen extends StatefulWidget {
  @override
  _RaceInfoScreenState createState() => _RaceInfoScreenState();
}

class _RaceInfoScreenState extends State<RaceInfoScreen> {
  Map<String, dynamic>? _info;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    fetchRaceInfo();
  }

  Future<void> fetchRaceInfo() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final response = await http.get(Uri.parse('http://127.0.0.1:5000/api/race-info'));
      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);
        setState(() {
          _info = data;
          _loading = false;
        });
      } else {
        setState(() {
          _error = 'Server error: ${response.statusCode}';
          _loading = false;
        });
      }
    } catch (e) {
      setState(() {
        _error = 'Error fetching data: $e';
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(child: Text(_error!, style: TextStyle(color: Colors.red)));
    }

    return SingleChildScrollView(
      padding: EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text(
            _info!["name"],
            style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.red[700]),
          ),
          SizedBox(height: 16),
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Image.network(
              _info!["track_map"],
              height: 220,
              width: double.infinity,
              fit: BoxFit.cover,
            ),
          ),
          SizedBox(height: 16),
          Text(
            "Circuit Length: ${_info!['circuit_length']}\nNumber of Laps: ${_info!['laps']}\nRace Distance: ${_info!['distance']}",
            style: TextStyle(color: Colors.white70, fontSize: 16),
            textAlign: TextAlign.center,
          ),
          SizedBox(height: 20),
          Text(
            _info!["description"],
            style: TextStyle(color: Colors.white, fontSize: 18),
            textAlign: TextAlign.justify,
          ),
          SizedBox(height: 30),
          Text(
            "Historical Highlights",
            style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.red[700]),
          ),
          SizedBox(height: 10),
          Text(
            _info!["highlights"],
            style: TextStyle(color: Colors.white70, fontSize: 16),
            textAlign: TextAlign.justify,
          ),
        ],
      ),
    );
  }
}
