import { useState, useEffect } from 'react'
import { useQuery } from 'react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { 
  Activity, 
  AlertTriangle, 
  Battery, 
  Brain, 
  Calendar, 
  Download,
  Filter,
  Heart,
  Plus,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Users,
  Zap,
  Eye,
  BarChart3,
  Stethoscope,
  Target,
  Shield
} from 'lucide-react'

// Hooks and contexts
import { useAuth } from '@contexts/AuthContext'
import { useWebSocket } from '@contexts/WebSocketContext'
import { useNotifications } from '@contexts/NotificationContext'
import { useTheme } from '@contexts/ThemeContext'

// API and services
import { api } from '@services/api'

// Components
import Card from '@components/common/Card'
import Button from '@components/common/Button'
import LoadingSpinner from '@components/common/LoadingSpinner'
import StatusCard from './components/StatusCard'
import SensorReadings from './components/SensorReadings'
import AlertsPanel from './components/AlertsPanel'
import MLPredictionCard from './components/MLPredictionCard'
import DeviceStatusGrid from './components/DeviceStatusGrid'
import RecentActivities from './components/RecentActivities'
import QuickActions from './components/QuickActions'
import PerformanceMetrics from './components/PerformanceMetrics'

// Utility functions
import { formatDate, formatTime } from '@utils/dateUtils'
import { calculateRiskLevel, getRiskColor } from '@utils/medicalUtils'

function Dashboard() {
  const [timeRange, setTimeRange] = useState('day')
  const [selectedPatient, setSelectedPatient] = useState(null)
  const [refreshing, setRefreshing] = useState(false)
  const [filters, setFilters] = useState({
    riskLevel: 'all',
    deviceStatus: 'all',
    alertType: 'all'
  })

  const { user } = useAuth()
  const { lastMessage, connectionStatus } = useWebSocket()
  const { showInfo } = useNotifications()
  const { currentTheme } = useTheme()

  // Fetch dashboard data
  const { data: dashboardData, isLoading: dashboardLoading, refetch: refetchDashboard } = useQuery(
    ['dashboard', user?.id, timeRange],
    () => api.analytics.getPatientAnalytics(user?.id, { timeRange }),
    {
      enabled: !!user?.id,
      staleTime: 5 * 60 * 1000,
      refetchInterval: 30 * 1000,
    }
  )

  // Fetch ML predictions
  const { data: mlPredictions, isLoading: mlLoading } = useQuery(
    ['ml-predictions', user?.id],
    () => api.ml.predictNeuropathyProgression(user?.id, {
      age: user?.age,
      diabetesDuration: user?.diabetesDuration,
      hba1c: user?.hba1c,
      recentTestResults: true
    }),
    {
      enabled: !!user?.id,
      staleTime: 10 * 60 * 1000,
    }
  )

  // Fetch device status
  const { data: deviceStatus, isLoading: deviceLoading } = useQuery(
    ['device-status', user?.id],
    () => api.device.getDevices({ patientId: user?.id }),
    {
      enabled: !!user?.id,
      staleTime: 2 * 60 * 1000,
      refetchInterval: 10 * 1000,
    }
  )

  // Fetch recent alerts
  const { data: alerts, isLoading: alertsLoading } = useQuery(
    ['alerts', user?.id],
    () => api.alert.getAlerts({ patientId: user?.id, limit: 10 }),
    {
      enabled: !!user?.id,
      staleTime: 30 * 1000,
      refetchInterval: 60 * 1000,
    }
  )

  // Fetch system health
  const { data: systemHealth } = useQuery(
    ['system-health'],
    () => api.admin.getSystemHealth(),
    {
      enabled: user?.role === 'ADMIN',
      staleTime: 60 * 1000,
      refetchInterval: 120 * 1000,
    }
  )

  // Handle real-time updates
  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'device_data' || lastMessage.type === 'test_result') {
        refetchDashboard()
      }
    }
  }, [lastMessage, refetchDashboard])

  // Handle manual refresh
  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      await refetchDashboard()
      showInfo('Dashboard refreshed successfully')
    } catch (error) {
      console.error('Error refreshing dashboard:', error)
    } finally {
      setRefreshing(false)
    }
  }

  // Export data functionality
  const handleExportData = async () => {
    try {
      const response = await api.analytics.getPatientAnalytics(user?.id, {
        timeRange,
        format: 'csv',
        includeMLPredictions: true
      })
      
      const blob = new Blob([response.data], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `patient-data-${user?.id}-${formatDate(new Date())}.csv`
      a.click()
      window.URL.revokeObjectURL(url)
      
      showInfo('Data exported successfully')
    } catch (error) {
      console.error('Error exporting data:', error)
    }
  }

  // Calculate dashboard metrics
  const getOverviewMetrics = () => {
    if (!dashboardData) return []

    const { patient, metrics, testResults } = dashboardData

    return [
      {
        label: 'Risk Score',
        value: metrics?.riskScore || 0,
        icon: Activity,
        change: metrics?.riskScoreChange || 0,
        changeType: (metrics?.riskScoreChange || 0) > 0 ? 'negative' : 'positive',
        info: 'Based on recent test results and ML predictions',
        color: getRiskColor(metrics?.riskScore || 0)
      },
      {
        label: 'Neuropathy Severity',
        value: metrics?.neuropathySeverity || 'Normal',
        icon: Brain,
        change: metrics?.neuropathyChange || 'Stable',
        changeType: 'neutral',
        info: 'Current neuropathy assessment level',
        color: getRiskColor(metrics?.neuropathyScore || 0)
      },
      {
        label: 'Test Compliance',
        value: `${metrics?.complianceRate || 0}%`,
        icon: Target,
        change: metrics?.complianceChange || 0,
        changeType: (metrics?.complianceChange || 0) > 0 ? 'positive' : 'negative',
        info: 'Adherence to scheduled testing',
        color: metrics?.complianceRate > 80 ? 'green' : metrics?.complianceRate > 60 ? 'yellow' : 'red'
      },
      {
        label: 'Device Status',
        value: deviceStatus?.data?.filter(d => d.status === 'ACTIVE')?.length || 0,
        icon: Shield,
        change: 'Active',
        changeType: 'positive',
        info: 'Connected smart shoe devices',
        color: 'blue'
      }
    ]
  }

  // Get recent activities
  const getRecentActivities = () => {
    if (!dashboardData) return []

    const activities = []
    
    // Add test results
    if (dashboardData.testResults) {
      dashboardData.testResults.slice(0, 5).forEach(test => {
        activities.push({
          id: test.id,
          type: 'test',
          title: 'Test Completed',
          description: `${test.testType} test completed`,
          timestamp: test.completedAt,
          icon: Stethoscope,
          color: 'blue'
        })
      })
    }

    // Add ML predictions
    if (mlPredictions) {
      activities.push({
        id: `ml-${Date.now()}`,
        type: 'prediction',
        title: 'ML Prediction Updated',
        description: `Risk level: ${mlPredictions.risk_level}`,
        timestamp: new Date().toISOString(),
        icon: Brain,
        color: getRiskColor(mlPredictions.prediction)
      })
    }

    // Add device events
    if (deviceStatus?.data) {
      deviceStatus.data.forEach(device => {
        if (device.lastActivity) {
          activities.push({
            id: `device-${device.id}`,
            type: 'device',
            title: 'Device Activity',
            description: `${device.name} - ${device.status}`,
            timestamp: device.lastActivity,
            icon: Activity,
            color: device.status === 'ACTIVE' ? 'green' : 'yellow'
          })
        }
      })
    }

    return activities.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)).slice(0, 8)
  }

  // Loading state
  if (dashboardLoading || mlLoading || deviceLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" text="Loading dashboard..." />
      </div>
    )
  }

  const overviewMetrics = getOverviewMetrics()
  const recentActivities = getRecentActivities()

  return (
    <div className="space-y-6">
      <Helmet>
        <title>Dashboard - Smart Shoe Monitor</title>
        <meta name="description" content="Monitor your diabetic neuropathy progress with real-time analytics" />
      </Helmet>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col lg:flex-row lg:items-center lg:justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Dashboard
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Welcome back, {user?.firstName}. Here's your health overview.
          </p>
        </div>
        
        <div className="mt-4 lg:mt-0 flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={refreshing}
            className="flex items-center gap-2"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={handleExportData}
            className="flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export Data
          </Button>
          
          <Button
            size="sm"
            className="flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New Test
          </Button>
        </div>
      </motion.div>

      {/* Connection Status */}
      {connectionStatus !== 'connected' && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4"
        >
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
            <span className="text-sm text-yellow-800 dark:text-yellow-200">
              Real-time connection {connectionStatus}. Some features may be limited.
            </span>
          </div>
        </motion.div>
      )}

      {/* Overview Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
      >
        {overviewMetrics.map((metric, index) => (
          <StatusCard
            key={index}
            {...metric}
            index={index}
          />
        ))}
      </motion.div>

      {/* Time Range Selector */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="flex items-center justify-between"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Time Range:
          </span>
          <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
            {['day', 'week', 'month'].map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                  timeRange === range
                    ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                {range.charAt(0).toUpperCase() + range.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <Button
          variant="outline"
          size="sm"
          className="flex items-center gap-2"
        >
          <Filter className="w-4 h-4" />
          Filter
        </Button>
      </motion.div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* ML Predictions */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="lg:col-span-4"
        >
          <MLPredictionCard
            predictions={mlPredictions}
            loading={mlLoading}
          />
        </motion.div>

        {/* Sensor Readings */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="lg:col-span-8"
        >
          <Card title="Sensor Readings" className="h-full">
            <SensorReadings 
              timeRange={timeRange}
              data={dashboardData?.sensorData}
            />
          </Card>
        </motion.div>
      </div>

      {/* Second Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Device Status */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
        >
          <DeviceStatusGrid
            devices={deviceStatus?.data || []}
            loading={deviceLoading}
          />
        </motion.div>

        {/* Recent Activities */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <RecentActivities
            activities={recentActivities}
          />
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.7 }}
        >
          <QuickActions
            user={user}
            onRefresh={handleRefresh}
            onExport={handleExportData}
          />
        </motion.div>
      </div>

      {/* Alerts Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
      >
        <Card title="Recent Alerts" className="overflow-hidden">
          <AlertsPanel
            alerts={alerts?.data || []}
            loading={alertsLoading}
          />
        </Card>
      </motion.div>

      {/* Performance Metrics for Admins */}
      {user?.role === 'ADMIN' && systemHealth && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
        >
          <PerformanceMetrics
            systemHealth={systemHealth}
          />
        </motion.div>
      )}
    </div>
  )
}

export default Dashboard