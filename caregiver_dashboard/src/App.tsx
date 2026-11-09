import { useState, useEffect, useRef } from 'react';
import './index.css';
import { 
  Activity, MapPin, AlertTriangle, Clock, Shield, Video, Bell,
  Camera, Circle, Cpu, HardDrive, Users,
  Heart, Navigation, Eye, Brain, Home, Settings
} from 'lucide-react';

// ==================== INTERFACES ====================
interface VisionContext {
  mode: string;
  torch_on: boolean;
  detected_objects: string[];
  closest_obstacle: string | null;
  wall_ahead: boolean;
  fps?: number;
}

interface IndoorNavigation {
  state: string;
  destination: string | null;
  confidence: string;
  distance_remaining: number | null;
  safe_directions: number;
  hold_reason: string | null;
}

interface SpatialMemory {
  name: string;
  room_id?: number;
  landmark_id?: number;
}

interface FallDetectorData {
  ax: number;
  ay: number;
  az: number;
  acceleration: number;
  pitch: number;
  roll: number;
  falls: number;
  state: 'NORMAL' | 'POSSIBLE_FALL' | 'FREEFALL' | 'IMPACT' | 'IMPACT_DETECTED' | 'FALL_CONFIRMED' | string;
  source?: 'HARDWARE' | 'SIMULATION' | 'UNAVAILABLE' | string;
  mpu?: string;
}

interface LogEntry {
  id: number;
  time: string;
  type: 'speech' | 'alert' | 'system' | 'navigation' | 'fall' | 'emergency';
  message: string;
  priority?: 'low' | 'medium' | 'high' | 'critical';
}

interface Location {
  lat: number;
  lng: number;
  city: string;
  region: string;
  country: string;
}

interface SystemMetrics {
  uptime: number;
  cpu_usage?: number;
  memory_usage?: number;
  temperature?: number;
}

const getBaseHost = () => {
  if (typeof window !== 'undefined' && window.location.hostname) {
    return window.location.hostname;
  }
  return 'localhost';
};

const getBasePort = () => {
  if (typeof window !== 'undefined' && window.location.port) {
    return window.location.port;
  }
  return '8000';
};

const API = import.meta.env.VITE_API_URL || `${typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'https:' : 'http:'}//${getBaseHost()}:${getBasePort()}`;
const WS  = import.meta.env.VITE_WS_URL || `${typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${getBaseHost()}:${getBasePort()}/ws`;
const VIDEO_FEED = `${API}/video_feed`;

function App() {
  const [activeView, setActiveView] = useState<'overview' | 'live-safety' | 'navigation' | 'system-health'>('overview');
  const [vision, setVision] = useState<VisionContext | null>(null);
  const [indoorNav, setIndoorNav] = useState<IndoorNavigation | null>(null);
  const [spatialMemory, setSpatialMemory] = useState<SpatialMemory[]>([]);
  const [fallData, setFallData] = useState<FallDetectorData | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [location, setLocation] = useState<Location | null>(null);
  const [wsState, setWsState] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [controllerConnected, setControllerConnected] = useState(false);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({ uptime: 0 });
  const [emergencyMode, setEmergencyMode] = useState(false);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fallPollInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const uptimeInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTime = useRef(Date.now());

  const addLog = (type: LogEntry['type'], message: string, priority: LogEntry['priority'] = 'medium') => {
    setLogs(prev => [
      {
        id: Date.now() + Math.random(),
        time: new Date().toLocaleTimeString(),
        type,
        message,
        priority
      },
      ...prev
    ].slice(0, 100));
  };

  const fetchFallData = async () => {
    try {
      const response = await fetch(`${API}/imu`, { signal: AbortSignal.timeout(2000) });
      const data = await response.json();
      if (!data || data.source === 'UNAVAILABLE') {
        setControllerConnected(false);
        setFallData(null);
        return;
      }
      const prevState = fallData?.state;
      setFallData(data);
      setControllerConnected(true);
      
      if (data.state === 'FALL_CONFIRMED' && prevState !== 'FALL_CONFIRMED') {
        const tag = data.source === 'SIMULATION' ? '[SIMULATION] ' : '';
        addLog('fall', `🚨 ${tag}FALL DETECTED! Total falls: ${data.falls}`, 'critical');
        setEmergencyMode(true);
      }
    } catch {
      if (controllerConnected) setControllerConnected(false);
      setFallData(null);
    }
  };

  const fetchLocation = () => {
    fetch(`${API}/location`)
      .then(r => r.json())
      .then(loc => setLocation(loc))
      .catch(() => {});
  };

  const connect = () => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
    }
    setWsState('connecting');
    const ws = new WebSocket(WS);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsState('connected');
      addLog('system', '✅ Dashboard connected', 'low');
      fetchLocation();
    };

    ws.onclose = () => {
      setWsState('disconnected');
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => ws.close();

    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        if (parsed.type === 'vision') setVision(parsed.data);
        else if (parsed.type === 'speech') addLog('speech', parsed.data.text, 'low');
        else if (parsed.type === 'indoor_navigation') setIndoorNav(parsed.data);
        else if (parsed.type === 'room_labeled') {
          addLog('system', `🏠 ${parsed.data.name}`, 'low');
          setSpatialMemory(prev => [...prev, parsed.data]);
        }
      } catch (e) {}
    };
  };

  useEffect(() => {
    connect();
    fetchFallData();
    fallPollInterval.current = setInterval(fetchFallData, 200);
    uptimeInterval.current = setInterval(() => {
      setSystemMetrics(prev => ({ ...prev, uptime: Math.floor((Date.now() - startTime.current) / 1000) }));
    }, 1000);
    
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (fallPollInterval.current) clearInterval(fallPollInterval.current);
      if (uptimeInterval.current) clearInterval(uptimeInterval.current);
      if (wsRef.current) { wsRef.current.onclose = null; wsRef.current.close(); }
    };
  }, []);

  const formatUptime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  };

  // Emergency Mode
  if (emergencyMode || fallData?.state === 'FALL_CONFIRMED') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-red-900 to-red-800 flex items-center justify-center p-6">
        <div className="max-w-2xl w-full bg-white rounded-2xl shadow-2xl p-12 text-center">
          <AlertTriangle className="w-24 h-24 text-red-600 mx-auto mb-6 animate-bounce" />
          <h1 className="text-5xl font-bold text-red-600 mb-4">EMERGENCY</h1>
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Fall Detected</h2>
          <div className="bg-red-50 rounded-xl p-6 mb-6">
            <div className="text-6xl font-bold text-red-600">{fallData?.falls || 0}</div>
            <div className="text-gray-600">Total Falls</div>
          </div>
          <button
            onClick={() => setEmergencyMode(false)}
            className="px-8 py-4 bg-red-600 hover:bg-red-700 text-white rounded-xl font-bold text-lg"
          >
            Acknowledge & Return
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center gap-3 mb-1">
            <div className="w-10 h-10 rounded-lg bg-blue-600 flex items-center justify-center">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">SURDAS</h1>
              <span className="text-xs text-gray-500 uppercase tracking-wide">Caregiver Hub</span>
            </div>
          </div>
        </div>

        <nav className="flex-1 p-3 space-y-1">
          {[
            { id: 'overview', label: 'Overview', icon: Home },
            { id: 'live-safety', label: 'Live Safety', icon: Shield },
            { id: 'navigation', label: 'Navigation', icon: Navigation },
            { id: 'system-health', label: 'System Health', icon: Cpu },
          ].map(item => (
            <button
              key={item.id}
              onClick={() => setActiveView(item.id as any)}
              className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all text-sm font-medium ${
                activeView === item.id
                  ? 'bg-blue-50 text-blue-600'
                  : 'text-gray-700 hover:bg-gray-50'
              }`}
            >
              <item.icon className="w-5 h-5" />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="p-4 border-t border-gray-200 space-y-3">
          <div className="bg-gray-50 px-3 py-2 rounded-lg flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${wsState === 'connected' ? 'bg-green-500' : 'bg-gray-300'}`} />
              <span className="text-sm text-gray-700">System</span>
            </div>
            <span className="text-xs text-gray-500">{wsState === 'connected' ? 'Online' : 'Offline'}</span>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
              <Users className="w-4 h-4 text-blue-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900">Dr. Elena Vance</p>
              <p className="text-xs text-gray-500">Caregiver</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col bg-gray-50">
        {/* Top Bar */}
        <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 bg-gray-100 px-3 py-1.5 rounded-full">
              <div className={`w-2 h-2 rounded-full ${
                wsState === 'connected' ? 'bg-green-500 animate-pulse' : 
                wsState === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
              <span className="text-sm text-gray-700 font-medium">
                {wsState === 'connected' ? 'Connected' : wsState === 'connecting' ? 'Connecting' : 'Offline'}
              </span>
            </div>

            {location && (
              <div className="flex items-center gap-2 text-gray-600">
                <MapPin className="w-4 h-4" />
                <span className="text-sm">{location.city}, {location.region}</span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={async () => {
                try {
                  await fetch(`${API}/emergency_stop`, { method: 'POST' });
                  addLog('system', '🛑 EMERGENCY STOP TRIGGERED FROM DASHBOARD', 'critical');
                } catch {}
              }}
              className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white font-bold text-xs rounded-lg flex items-center gap-1 shadow-sm transition-colors"
              title="Immediate deterministic hardware stop"
            >
              <AlertTriangle className="w-3.5 h-3.5" />
              EMERGENCY STOP
            </button>
            <div className="text-sm text-gray-500">{new Date().toLocaleTimeString()}</div>
            <button className="relative p-2 hover:bg-gray-100 rounded-lg">
              <Bell className="w-5 h-5 text-gray-600" />
              {logs.filter(l => l.priority === 'critical').length > 0 && (
                <div className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
              )}
            </button>
            <button className="p-2 hover:bg-gray-100 rounded-lg">
              <Settings className="w-5 h-5 text-gray-600" />
            </button>
          </div>
        </header>

        {/* Content Area */}
        <main className="flex-1 overflow-auto p-6">
          {activeView === 'overview' && (
            <div className="max-w-7xl mx-auto space-y-6">
              {/* Status Banner */}
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-full bg-green-100 flex items-center justify-center">
                    <Shield className="w-7 h-7 text-green-600" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                      <h2 className="text-xl font-bold text-green-900">User is Safe</h2>
                    </div>
                    <p className="text-green-700 text-sm">All systems operational • Last updated 2s ago</p>
                  </div>
                </div>
              </div>

              {/* Quick Stats */}
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-gray-500 uppercase">SURDAS</span>
                    <Activity className={`w-4 h-4 ${wsState === 'connected' ? 'text-green-600' : 'text-red-600'}`} />
                  </div>
                  <div className="text-2xl font-bold text-gray-900">{wsState === 'connected' ? 'Online' : 'Offline'}</div>
                </div>

                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-gray-500 uppercase">Camera</span>
                    <Video className="w-4 h-4 text-green-600" />
                  </div>
                  <div className="text-2xl font-bold text-gray-900">Active</div>
                </div>

                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-gray-500 uppercase">Navigation</span>
                    <Navigation className="w-4 h-4 text-blue-600" />
                  </div>
                  <div className="text-2xl font-bold text-gray-900">{indoorNav?.state || 'Idle'}</div>
                </div>

                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-gray-500 uppercase">Falls Detected</span>
                    <Heart className="w-4 h-4 text-purple-600" />
                  </div>
                  <div className="text-2xl font-bold text-gray-900">{fallData?.falls || 0}</div>
                </div>
              </div>

              {/* Main Grid */}
              <div className="grid grid-cols-3 gap-6">
                {/* Large Video Feed */}
                <div className="col-span-2 bg-white border border-gray-200 rounded-xl overflow-hidden">
                  <div className="bg-gray-50 border-b border-gray-200 px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Video className="w-5 h-5 text-gray-700" />
                      <div>
                        <h3 className="text-base font-semibold text-gray-900">Live Camera Feed</h3>
                        <p className="text-xs text-gray-500">Real-time vision • {vision?.fps?.toFixed(0) || 0} FPS</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 bg-red-50 px-3 py-1 rounded-full">
                      <Circle className="w-2 h-2 fill-red-500 text-red-500" />
                      <span className="text-xs font-bold text-red-600">LIVE</span>
                    </div>
                  </div>

                  <div className="relative bg-black aspect-video">
                    <img 
                      src={VIDEO_FEED} 
                      alt="Feed" 
                      className="w-full h-full object-contain" 
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.style.display = 'none';
                      }}
                    />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center text-gray-400">
                        <Camera className="w-16 h-16 mx-auto mb-2 opacity-20" />
                        <p className="text-sm">Loading camera feed...</p>
                      </div>
                    </div>
                    <div className="absolute bottom-4 left-4 bg-black/70 backdrop-blur px-3 py-1.5 rounded-lg">
                      <span className="text-white text-sm">Mode: {vision?.mode || 'IDLE'}</span>
                    </div>
                  </div>

                  <div className="bg-gray-50 border-t border-gray-200 px-6 py-4">
                    <div className="grid grid-cols-3 gap-6">
                      <div>
                        <div className="text-xs text-gray-500 mb-1">Objects Detected</div>
                        <div className="text-2xl font-bold text-gray-900">{vision?.detected_objects?.length || 0}</div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-500 mb-1">Closest Obstacle</div>
                        <div className="text-sm font-medium text-gray-900">{vision?.closest_obstacle || 'Clear'}</div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-500 mb-1">Wall Ahead</div>
                        <div className={`text-sm font-bold ${vision?.wall_ahead ? 'text-red-600' : 'text-green-600'}`}>
                          {vision?.wall_ahead ? '⚠️ Yes' : '✅ Clear'}
                        </div>
                      </div>
                    </div>

                    {vision && vision.detected_objects.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-gray-200">
                        <div className="text-xs text-gray-500 mb-2">Detected Objects:</div>
                        <div className="flex flex-wrap gap-2">
                          {[...new Set(vision.detected_objects)].slice(0, 8).map((obj, i) => (
                            <span key={i} className="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded font-medium">
                              {obj}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Right Column */}
                <div className="space-y-6">
                  {/* System Health */}
                  <div className="bg-white border border-gray-200 rounded-xl p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <Activity className="w-5 h-5 text-gray-700" />
                      <h3 className="text-base font-semibold text-gray-900">System Health</h3>
                    </div>
                    <div className="space-y-3">
                      {[
                        { label: 'Uptime', value: formatUptime(systemMetrics.uptime), icon: Clock },
                        { label: 'CPU', value: `${systemMetrics.cpu_usage?.toFixed(0) || '--'}%`, icon: Cpu },
                        { label: 'Memory', value: `${systemMetrics.memory_usage?.toFixed(0) || '--'}%`, icon: HardDrive },
                      ].map((item, i) => (
                        <div key={i} className="flex items-center justify-between bg-gray-50 px-3 py-2.5 rounded-lg">
                          <div className="flex items-center gap-2">
                            <item.icon className="w-4 h-4 text-gray-600" />
                            <span className="text-sm text-gray-700">{item.label}</span>
                          </div>
                          <span className="text-sm font-bold text-gray-900">{item.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Fall & Motion Detector */}
                  <div className="bg-white border border-gray-200 rounded-xl p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <Heart className="w-5 h-5 text-gray-700" />
                      <h3 className="text-base font-semibold text-gray-900">Fall Detector</h3>
                      {fallData?.source === 'SIMULATION' ? (
                        <span className="ml-auto text-xs font-semibold px-2 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-300">
                          SIMULATION
                        </span>
                      ) : controllerConnected && fallData?.source === 'HARDWARE' ? (
                        <span className="ml-auto text-xs font-semibold px-2 py-0.5 rounded bg-green-100 text-green-800 border border-green-300">
                          STM32 HARDWARE
                        </span>
                      ) : (
                        <span className="ml-auto text-xs font-medium px-2 py-0.5 rounded bg-gray-100 text-gray-600">
                          UNAVAILABLE
                        </span>
                      )}
                    </div>
                    {controllerConnected && fallData ? (
                      <>
                        <div className={`p-4 rounded-lg text-center mb-4 ${
                          fallData.state === 'NORMAL' ? 'bg-green-50 border border-green-200' :
                          fallData.state === 'POSSIBLE_FALL' || fallData.state === 'TILT_WARNING' ? 'bg-yellow-50 border border-yellow-200' :
                          'bg-red-50 border border-red-200'
                        }`}>
                          <div className={`text-base font-bold ${
                            fallData.state === 'NORMAL' ? 'text-green-900' :
                            fallData.state === 'POSSIBLE_FALL' || fallData.state === 'TILT_WARNING' ? 'text-yellow-900' :
                            'text-red-900'
                          }`}>
                            {fallData.state.replace('_', ' ')}
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="bg-gray-50 p-3 rounded-lg text-center">
                            <div className="text-xl font-bold text-gray-900">{typeof fallData.acceleration === 'number' ? fallData.acceleration.toFixed(1) : '0.0'}</div>
                            <div className="text-xs text-gray-500">m/s²</div>
                          </div>
                          <div className="bg-gray-50 p-3 rounded-lg text-center">
                            <div className="text-xl font-bold text-gray-900">{fallData.falls}</div>
                            <div className="text-xs text-gray-500">Total Falls</div>
                          </div>
                        </div>
                      </>
                    ) : (
                      <p className="text-xs text-gray-500 italic py-2">
                        No physical IMU sensor attached to Arduino UNO Q. Monitoring unavailable.
                      </p>
                    )}
                  </div>

                  {/* Quick Actions */}
                  <div className="bg-white border border-gray-200 rounded-xl p-6">
                    <h3 className="text-base font-semibold text-gray-900 mb-4">Quick Stats</h3>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Eye className="w-4 h-4 text-gray-600" />
                          <span className="text-sm text-gray-700">Vision Objects</span>
                        </div>
                        <span className="text-sm font-bold text-gray-900">{vision?.detected_objects?.length || 0}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Brain className="w-4 h-4 text-gray-600" />
                          <span className="text-sm text-gray-700">Spatial Memory</span>
                        </div>
                        <span className="text-sm font-bold text-gray-900">{spatialMemory.length}</span>
                      </div>
                      {location && (
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <MapPin className="w-4 h-4 text-gray-600" />
                            <span className="text-sm text-gray-700">Location</span>
                          </div>
                          <span className="text-sm font-bold text-gray-900">{location.city}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Activity Logs */}
              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <h3 className="text-base font-semibold text-gray-900 mb-4">Recent Activity</h3>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {logs.slice(0, 10).map(log => (
                    <div key={log.id} className={`p-3 rounded-lg border ${
                      log.type === 'fall' ? 'bg-red-50 border-red-200' :
                      log.type === 'alert' ? 'bg-yellow-50 border-yellow-200' :
                      log.type === 'navigation' ? 'bg-blue-50 border-blue-200' :
                      'bg-gray-50 border-gray-200'
                    }`}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-gray-500 font-mono">{log.time}</span>
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                          log.type === 'fall' ? 'bg-red-100 text-red-700' :
                          log.type === 'alert' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-blue-100 text-blue-700'
                        }`}>
                          {log.type.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-sm text-gray-900">{log.message}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeView === 'live-safety' && (
            <div className="max-w-7xl mx-auto space-y-6">
              <h2 className="text-2xl font-bold text-gray-900">Live Safety Monitor</h2>
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                <div className="aspect-video bg-black relative">
                  <img src={VIDEO_FEED} alt="Feed" className="w-full h-full object-contain" />
                </div>
              </div>
            </div>
          )}

          {activeView === 'navigation' && (
            <div className="max-w-7xl mx-auto space-y-6">
              <h2 className="text-2xl font-bold text-gray-900">Indoor Navigation</h2>
              {indoorNav && (
                <div className="bg-white border border-gray-200 rounded-xl p-6">
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <div className="text-sm text-gray-500 mb-1">State</div>
                      <div className="text-xl font-bold text-gray-900">{indoorNav.state}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-500 mb-1">Destination</div>
                      <div className="text-xl font-bold text-gray-900">{indoorNav.destination || 'None'}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeView === 'system-health' && (
            <div className="max-w-7xl mx-auto space-y-6">
              <h2 className="text-2xl font-bold text-gray-900">System Health</h2>
              <div className="grid grid-cols-2 gap-6">
                <div className="bg-white border border-gray-200 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-gray-600">CPU Usage</span>
                        <span className="text-sm font-bold text-gray-900">{systemMetrics.cpu_usage?.toFixed(0) || 0}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${systemMetrics.cpu_usage || 0}%` }} />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-gray-600">Memory Usage</span>
                        <span className="text-sm font-bold text-gray-900">{systemMetrics.memory_usage?.toFixed(0) || 0}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-purple-600 h-2 rounded-full" style={{ width: `${systemMetrics.memory_usage || 0}%` }} />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
