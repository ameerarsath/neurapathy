import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Server, 
  Database, 
  Cpu, 
  MemoryStick, 
  HardDrive, 
  Network, 
  AlertTriangle, 
  CheckCircle,
  Activity,
  TrendingUp,
  TrendingDown,
  Clock,
  Users,
  BarChart3,
  Zap,
  Eye,
  RefreshCw
} from 'lucide-react'
import Card from '@components/common/Card'
import Button from '@components/common/Button'

function PerformanceMetrics({ systemHealth }) {
  const [selectedMetric, setSelectedMetric] = useState(null)
  const [timeRange, setTimeRange] = useState('1h')

  if (!systemHealth) {
    return (
      <Card title="System Performance" className="h-full">
        <div className="flex items-center justify-center h-40 text-gray-500 dark:text-gray-400">
          <Server className="w-8 h-8 mb-2" />
          <p className="text-sm">System health data unavailable</p>
        </div>
      </Card>
    )
  }

  const getStatusColor = (status, value, threshold) => {
    if (status === 'UP' || status === 'HEALTHY') {
      return 'text-green-600 bg-green-100 dark:bg-green-900/20'
    } else if (status === 'DOWN' || status === 'UNHEALTHY') {
      return 'text-red-600 bg-red-100 dark:bg-red-900/20'
    } else if (value && threshold && value > threshold) {
      return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20'
    }
    return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20'
  }

  const getStatusIcon = (status, value, threshold) => {
    if (status === 'UP' || status === 'HEALTHY') {
      return <CheckCircle className="w-4 h-4" />
    } else if (status === 'DOWN' || status === 'UNHEALTHY') {
      return <AlertTriangle className="w-4 h-4" />
    } else if (value && threshold && value > threshold) {
      return <AlertTriangle className="w-4 h-4" />
    }
    return <Activity className="w-4 h-4" />
  }

  const metrics = [
    {
      id: 'api-response',
      label: 'API Response',
      value: systemHealth.apiResponseTime || '45ms',
      status: systemHealth.apiStatus || 'UP',
      icon: Zap,
      threshold: 100,
      trend: 'down',
      change: '-12ms',
      description: 'Average API response time'
    },
    {
      id: 'database',
      label: 'Database',
      value: systemHealth.databaseConnections || '8/20',
      status: systemHealth.databaseStatus || 'UP',
      icon: Database,
      threshold: 15,
      trend: 'stable',
      change: '+1',
      description: 'Active database connections'
    },
    {
      id: 'cpu',
      label: 'CPU Usage',
      value: systemHealth.cpuUsage || '45%',
      status: 'HEALTHY',
      icon: Cpu,
      threshold: 80,
      trend: 'up',
      change: '+5%',
      description: 'Current CPU utilization'
    },
    {
      id: 'memory',
      label: 'Memory',
      value: systemHealth.memoryUsage || '2.1GB/4GB',
      status: 'HEALTHY',
      icon: MemoryStick,
      threshold: 85,
      trend: 'stable',
      change: '+0.1GB',
      description: 'Memory consumption'
    },
    {
      id: 'storage',
      label: 'Storage',
      value: systemHealth.storageUsage || '45GB/100GB',
      status: 'HEALTHY',
      icon: HardDrive,
      threshold: 90,
      trend: 'up',
      change: '+2GB',
      description: 'Disk space utilization'
    },
    {
      id: 'network',
      label: 'Network',
      value: systemHealth.networkLatency || '12ms',
      status: systemHealth.networkStatus || 'UP',
      icon: Network,
      threshold: 50,
      trend: 'down',
      change: '-3ms',
      description: 'Network latency'
    }
  ]

  const applicationMetrics = [
    {
      label: 'Active Users',
      value: systemHealth.activeUsers || '127',
      icon: Users,
      change: '+15',
      changeType: 'positive'
    },
    {
      label: 'Tests/Hour',
      value: systemHealth.testsPerHour || '23',
      icon: BarChart3,
      change: '+5',
      changeType: 'positive'
    },
    {
      label: 'Predictions/Min',
      value: systemHealth.predictionsPerMinute || '8',
      icon: Activity,
      change: '+2',
      changeType: 'positive'
    },
    {
      label: 'Error Rate',
      value: systemHealth.errorRate || '0.1%',
      icon: AlertTriangle,
      change: '-0.05%',
      changeType: 'positive'
    }
  ]

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'up': return <TrendingUp className="w-3 h-3 text-red-500" />
      case 'down': return <TrendingDown className="w-3 h-3 text-green-500" />
      default: return <Activity className="w-3 h-3 text-gray-500" />
    }
  }

  return (
    <Card 
      title="System Performance" 
      className="h-full"
      actions={
        <div className="flex items-center gap-2">
          <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
            {['1h', '6h', '24h'].map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                  timeRange === range
                    ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                    : 'text-gray-600 dark:text-gray-400'
                }`}
              >
                {range}
              </button>
            ))}
          </div>
          <Button variant="ghost" size="sm">
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      }
    >
      <div className="space-y-6">
        {/* System Health Metrics */}
        <div>
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Infrastructure Health
          </h4>
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
            {metrics.map((metric, index) => {
              const Icon = metric.icon
              const isSelected = selectedMetric === metric.id
              
              return (
                <motion.div
                  key={metric.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className={`p-3 border rounded-lg cursor-pointer transition-all ${
                    isSelected
                      ? 'border-blue-300 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                  }`}
                  onClick={() => setSelectedMetric(isSelected ? null : metric.id)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <Icon className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                    <span className={`px-2 py-1 text-xs font-medium rounded-full flex items-center gap-1 ${getStatusColor(metric.status)}`}>
                      {getStatusIcon(metric.status)}
                      {metric.status}
                    </span>
                  </div>
                  
                  <div className="mb-2">
                    <div className="text-lg font-semibold text-gray-900 dark:text-white">
                      {metric.value}
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      {metric.label}
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-1">
                      {getTrendIcon(metric.trend)}
                      <span className="text-gray-500">{metric.change}</span>
                    </div>
                    <Clock className="w-3 h-3 text-gray-400" />
                  </div>

                  {/* Detailed View */}
                  {isSelected && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700"
                    >
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {metric.description}
                      </p>
                      <div className="mt-2 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1">
                        <div 
                          className="bg-blue-600 h-1 rounded-full transition-all duration-1000"
                          style={{ width: '75%' }}
                        />
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              )
            })}
          </div>
        </div>

        {/* Application Metrics */}
        <div>
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Application Metrics
          </h4>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {applicationMetrics.map((metric, index) => {
              const Icon = metric.icon
              
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + index * 0.05 }}
                  className="p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg"
                >
                  <div className="flex items-center justify-between mb-2">
                    <Icon className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                    <span className={`text-xs font-medium ${
                      metric.changeType === 'positive' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {metric.change}
                    </span>
                  </div>
                  
                  <div className="text-lg font-semibold text-gray-900 dark:text-white">
                    {metric.value}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {metric.label}
                  </div>
                </motion.div>
              )
            })}
          </div>
        </div>

        {/* Overall System Status */}
        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                All Systems Operational
              </span>
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
              <Clock className="w-3 h-3" />
              <span>Last updated: {new Date().toLocaleTimeString()}</span>
            </div>
          </div>
          
          <div className="mt-2 flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>Uptime: 99.9% (30 days)</span>
            <Button variant="ghost" size="xs">
              <Eye className="w-3 h-3 mr-1" />
              View Details
            </Button>
          </div>
        </div>
      </div>
    </Card>
  )
}

export default PerformanceMetrics