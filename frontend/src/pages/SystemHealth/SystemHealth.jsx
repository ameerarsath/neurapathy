import { useState } from 'react'
import { useQuery } from 'react-query'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { 
  Server, 
  Database, 
  Activity, 
  AlertTriangle,
  CheckCircle,
  Clock,
  Cpu,
  MemoryStick,
  HardDrive,
  Network,
  RefreshCw
} from 'lucide-react'

import { api } from '@services/api'
import Card from '@components/common/Card'
import Button from '@components/common/Button'
import LoadingSpinner from '@components/common/LoadingSpinner'

function SystemHealth() {
  const [autoRefresh, setAutoRefresh] = useState(true)

  const { data: healthData, isLoading, refetch } = useQuery(
    ['system-health'],
    () => api.admin.getSystemHealth(),
    {
      staleTime: 30 * 1000,
      refetchInterval: autoRefresh ? 30 * 1000 : false,
    }
  )

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" text="Loading system health..." />
      </div>
    )
  }

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'healthy': 
      case 'up': 
        return 'text-green-600 bg-green-100'
      case 'warning': 
        return 'text-yellow-600 bg-yellow-100'
      case 'critical': 
      case 'down': 
        return 'text-red-600 bg-red-100'
      default: 
        return 'text-gray-600 bg-gray-100'
    }
  }

  const services = [
    { name: 'API Server', status: 'up', uptime: '99.9%', responseTime: '45ms' },
    { name: 'Database', status: 'up', uptime: '99.8%', responseTime: '12ms' },
    { name: 'ML Service', status: 'up', uptime: '99.5%', responseTime: '150ms' },
    { name: 'Redis Cache', status: 'up', uptime: '99.9%', responseTime: '2ms' },
    { name: 'Message Queue', status: 'warning', uptime: '98.2%', responseTime: '25ms' },
  ]

  const metrics = [
    { label: 'CPU Usage', value: '45%', icon: Cpu, status: 'healthy' },
    { label: 'Memory', value: '2.1GB/4GB', icon: MemoryStick, status: 'healthy' },
    { label: 'Storage', value: '45GB/100GB', icon: HardDrive, status: 'healthy' },
    { label: 'Network', value: '12ms', icon: Network, status: 'healthy' },
  ]

  return (
    <div className="space-y-6">
      <Helmet>
        <title>System Health - Smart Shoe Monitor</title>
        <meta name="description" content="Monitor system health and performance metrics" />
      </Helmet>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col lg:flex-row lg:items-center lg:justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            System Health
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Monitor system performance and service status
          </p>
        </div>
        
        <div className="mt-4 lg:mt-0 flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={autoRefresh ? 'bg-green-50 text-green-700' : ''}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
            Auto Refresh
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </motion.div>

      {/* Overall Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-1 md:grid-cols-4 gap-4"
      >
        {metrics.map((metric, index) => {
          const Icon = metric.icon
          return (
            <Card key={index} className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                  <Icon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <div className="text-lg font-semibold text-gray-900 dark:text-white">
                    {metric.value}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    {metric.label}
                  </div>
                </div>
              </div>
            </Card>
          )
        })}
      </motion.div>

      {/* Services Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card title="Service Status">
          <div className="space-y-4">
            {services.map((service, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${
                    service.status === 'up' ? 'bg-green-500' : 
                    service.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                  }`} />
                  <div>
                    <div className="font-medium text-gray-900 dark:text-white">
                      {service.name}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      Uptime: {service.uptime}
                    </div>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(service.status)}`}>
                    {service.status.toUpperCase()}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    {service.responseTime}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </motion.div>

      {/* Recent Activity */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card title="Recent System Events">
          <div className="space-y-3">
            <div className="flex items-start gap-3 p-3 border-l-4 border-green-400 bg-green-50 dark:bg-green-900/20">
              <CheckCircle className="w-4 h-4 text-green-600 mt-0.5" />
              <div>
                <div className="text-sm font-medium text-gray-900 dark:text-white">
                  System backup completed successfully
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  2 minutes ago
                </div>
              </div>
            </div>
            
            <div className="flex items-start gap-3 p-3 border-l-4 border-yellow-400 bg-yellow-50 dark:bg-yellow-900/20">
              <AlertTriangle className="w-4 h-4 text-yellow-600 mt-0.5" />
              <div>
                <div className="text-sm font-medium text-gray-900 dark:text-white">
                  High memory usage detected on ML service
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  5 minutes ago
                </div>
              </div>
            </div>
            
            <div className="flex items-start gap-3 p-3 border-l-4 border-blue-400 bg-blue-50 dark:bg-blue-900/20">
              <Activity className="w-4 h-4 text-blue-600 mt-0.5" />
              <div>
                <div className="text-sm font-medium text-gray-900 dark:text-white">
                  Database optimization task completed
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  1 hour ago
                </div>
              </div>
            </div>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}

export default SystemHealth