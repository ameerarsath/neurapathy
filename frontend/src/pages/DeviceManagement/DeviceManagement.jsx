import { useState } from 'react'
import { useQuery } from 'react-query'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { 
  Smartphone, 
  Plus, 
  Settings, 
  Battery, 
  Wifi,
  Activity,
  AlertTriangle,
  CheckCircle,
  Trash2,
  Edit,
  Zap
} from 'lucide-react'

import { useAuth } from '@contexts/AuthContext'
import { api } from '@services/api'
import Card from '@components/common/Card'
import Button from '@components/common/Button'
import LoadingSpinner from '@components/common/LoadingSpinner'

function DeviceManagement() {
  const [selectedDevice, setSelectedDevice] = useState(null)
  const { user } = useAuth()

  const { data: devices, isLoading } = useQuery(
    ['devices', user?.id],
    () => api.device.getDevices({ patientId: user?.id }),
    {
      enabled: !!user?.id,
      staleTime: 2 * 60 * 1000,
    }
  )

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" text="Loading devices..." />
      </div>
    )
  }

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'active': return 'text-green-600 bg-green-100'
      case 'inactive': return 'text-red-600 bg-red-100'
      case 'calibrating': return 'text-yellow-600 bg-yellow-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  return (
    <div className="space-y-6">
      <Helmet>
        <title>Device Management - Smart Shoe Monitor</title>
        <meta name="description" content="Manage your smart shoe devices and monitoring equipment" />
      </Helmet>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col lg:flex-row lg:items-center lg:justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Device Management
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Manage your smart shoe devices and monitoring equipment
          </p>
        </div>
        
        <Button className="mt-4 lg:mt-0 flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Add Device
        </Button>
      </motion.div>

      {/* Device Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
      >
        {devices?.data?.map((device, index) => (
          <motion.div
            key={device.id}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card className="h-full hover:shadow-lg transition-shadow">
              <div className="p-6">
                {/* Device Header */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <Smartphone className="w-6 h-6 text-blue-600" />
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-white">
                        {device.name || `Smart Shoe ${device.id.slice(-4)}`}
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {device.model || 'SmartShoe v2.0'}
                      </p>
                    </div>
                  </div>
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(device.status)}`}>
                    {device.status || 'Active'}
                  </span>
                </div>

                {/* Device Stats */}
                <div className="space-y-3 mb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Battery className="w-4 h-4 text-gray-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Battery</span>
                    </div>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {device.batteryLevel || 85}%
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Wifi className="w-4 h-4 text-gray-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Signal</span>
                    </div>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {device.signalStrength || 'Strong'}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Activity className="w-4 h-4 text-gray-500" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">Last Active</span>
                    </div>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {device.lastActivity ? new Date(device.lastActivity).toLocaleTimeString() : '2 min ago'}
                    </span>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" className="flex-1">
                    <Settings className="w-4 h-4 mr-2" />
                    Configure
                  </Button>
                  <Button variant="outline" size="sm">
                    <Zap className="w-4 h-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <Edit className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </Card>
          </motion.div>
        ))}

        {/* Add Device Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
        >
          <Card className="h-full border-dashed border-2 border-gray-300 dark:border-gray-600 hover:border-blue-400 transition-colors cursor-pointer">
            <div className="p-6 flex flex-col items-center justify-center h-full text-center">
              <Plus className="w-8 h-8 text-gray-400 mb-3" />
              <h3 className="font-medium text-gray-700 dark:text-gray-300 mb-2">
                Add New Device
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Connect a new smart shoe or monitoring device
              </p>
            </div>
          </Card>
        </motion.div>
      </motion.div>

      {/* Device Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <Card title="Device Summary">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <CheckCircle className="w-8 h-8 text-green-600 mx-auto mb-2" />
              <div className="text-lg font-semibold text-green-600">
                {devices?.data?.filter(d => d.status === 'ACTIVE').length || 2}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Active Devices</div>
            </div>
            
            <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <Battery className="w-8 h-8 text-yellow-600 mx-auto mb-2" />
              <div className="text-lg font-semibold text-yellow-600">
                {devices?.data?.filter(d => (d.batteryLevel || 85) < 20).length || 0}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Low Battery</div>
            </div>
            
            <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <Activity className="w-8 h-8 text-blue-600 mx-auto mb-2" />
              <div className="text-lg font-semibold text-blue-600">
                {devices?.data?.reduce((sum, d) => sum + (d.testsToday || 0), 0) || 15}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Tests Today</div>
            </div>
            
            <div className="text-center p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <AlertTriangle className="w-8 h-8 text-red-600 mx-auto mb-2" />
              <div className="text-lg font-semibold text-red-600">
                {devices?.data?.filter(d => d.status === 'INACTIVE').length || 0}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Offline</div>
            </div>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}

export default DeviceManagement