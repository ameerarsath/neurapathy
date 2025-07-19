import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Smartphone, 
  Battery, 
  Wifi, 
  WifiOff, 
  Activity, 
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap,
  Settings,
  MoreVertical
} from 'lucide-react'
import Card from '@components/common/Card'
import Button from '@components/common/Button'
import LoadingSpinner from '@components/common/LoadingSpinner'

function DeviceStatusGrid({ devices, loading }) {
  const [selectedDevice, setSelectedDevice] = useState(null)

  if (loading) {
    return (
      <Card title="Device Status" className="h-full">
        <div className="flex items-center justify-center h-40">
          <LoadingSpinner size="md" text="Loading devices..." />
        </div>
      </Card>
    )
  }

  if (!devices || devices.length === 0) {
    return (
      <Card title="Device Status" className="h-full">
        <div className="flex flex-col items-center justify-center h-40 text-gray-500 dark:text-gray-400">
          <Smartphone className="w-8 h-8 mb-2" />
          <p className="text-sm">No devices connected</p>
          <Button variant="outline" size="sm" className="mt-2">
            Add Device
          </Button>
        </div>
      </Card>
    )
  }

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'active': 
      case 'connected': 
        return 'text-green-600 bg-green-100 dark:bg-green-900/20'
      case 'inactive': 
      case 'disconnected': 
        return 'text-red-600 bg-red-100 dark:bg-red-900/20'
      case 'calibrating': 
      case 'syncing': 
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20'
      case 'low_battery': 
        return 'text-orange-600 bg-orange-100 dark:bg-orange-900/20'
      default: 
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20'
    }
  }

  const getStatusIcon = (status) => {
    switch (status?.toLowerCase()) {
      case 'active': 
      case 'connected': 
        return <CheckCircle className="w-3 h-3" />
      case 'inactive': 
      case 'disconnected': 
        return <WifiOff className="w-3 h-3" />
      case 'calibrating': 
      case 'syncing': 
        return <Clock className="w-3 h-3" />
      case 'low_battery': 
        return <Battery className="w-3 h-3" />
      default: 
        return <AlertTriangle className="w-3 h-3" />
    }
  }

  const getBatteryLevel = (device) => {
    return device.batteryLevel || device.battery_level || 50
  }

  const getSignalStrength = (device) => {
    return device.signalStrength || device.signal_strength || 75
  }

  return (
    <Card 
      title="Device Status" 
      className="h-full"
      actions={
        <Button variant="ghost" size="sm">
          <Settings className="w-4 h-4" />
        </Button>
      }
    >
      <div className="space-y-3">
        {devices.map((device, index) => (
          <motion.div
            key={device.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`p-3 border rounded-lg cursor-pointer transition-all ${
              selectedDevice?.id === device.id
                ? 'border-blue-300 bg-blue-50 dark:bg-blue-900/20'
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
            }`}
            onClick={() => setSelectedDevice(selectedDevice?.id === device.id ? null : device)}
          >
            {/* Device Header */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Smartphone className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {device.name || `Device ${device.id.slice(-4)}`}
                </span>
              </div>
              <div className="flex items-center gap-1">
                <span className={`px-2 py-1 text-xs font-medium rounded-full flex items-center gap-1 ${getStatusColor(device.status)}`}>
                  {getStatusIcon(device.status)}
                  {device.status || 'Unknown'}
                </span>
                <Button variant="ghost" size="xs">
                  <MoreVertical className="w-3 h-3" />
                </Button>
              </div>
            </div>

            {/* Device Metrics */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              {/* Battery */}
              <div className="flex items-center gap-1">
                <Battery className="w-3 h-3 text-gray-500" />
                <span className="text-gray-600 dark:text-gray-400">Battery:</span>
                <span className={`font-medium ${getBatteryLevel(device) < 20 ? 'text-red-600' : 'text-gray-900 dark:text-white'}`}>
                  {getBatteryLevel(device)}%
                </span>
              </div>

              {/* Signal */}
              <div className="flex items-center gap-1">
                <Wifi className="w-3 h-3 text-gray-500" />
                <span className="text-gray-600 dark:text-gray-400">Signal:</span>
                <span className="font-medium text-gray-900 dark:text-white">
                  {getSignalStrength(device)}%
                </span>
              </div>

              {/* Last Activity */}
              <div className="flex items-center gap-1 col-span-2">
                <Activity className="w-3 h-3 text-gray-500" />
                <span className="text-gray-600 dark:text-gray-400">Last seen:</span>
                <span className="font-medium text-gray-900 dark:text-white">
                  {device.lastActivity 
                    ? new Date(device.lastActivity).toLocaleTimeString()
                    : 'Never'
                  }
                </span>
              </div>
            </div>

            {/* Battery Progress Bar */}
            <div className="mt-2">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-gray-500">Battery Level</span>
                <span className={getBatteryLevel(device) < 20 ? 'text-red-600' : 'text-gray-600'}>
                  {getBatteryLevel(device)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${getBatteryLevel(device)}%` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                  className={`h-1.5 rounded-full ${
                    getBatteryLevel(device) < 20 
                      ? 'bg-red-500' 
                      : getBatteryLevel(device) < 50 
                      ? 'bg-yellow-500' 
                      : 'bg-green-500'
                  }`}
                />
              </div>
            </div>

            {/* Expanded Details */}
            {selectedDevice?.id === device.id && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700"
              >
                <div className="space-y-2 text-xs">
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <span className="text-gray-500">Model:</span>
                      <span className="ml-1 font-medium">{device.model || 'SmartShoe v2'}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Firmware:</span>
                      <span className="ml-1 font-medium">{device.firmware || '2.1.0'}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Temperature:</span>
                      <span className="ml-1 font-medium">{device.temperature || '23'}°C</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Tests Today:</span>
                      <span className="ml-1 font-medium">{device.testsToday || 0}</span>
                    </div>
                  </div>

                  {/* Quick Actions */}
                  <div className="flex gap-1 mt-3">
                    <Button variant="outline" size="xs" className="flex-1">
                      <Zap className="w-3 h-3 mr-1" />
                      Calibrate
                    </Button>
                    <Button variant="outline" size="xs" className="flex-1">
                      <Settings className="w-3 h-3 mr-1" />
                      Configure
                    </Button>
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        ))}

        {/* Summary Stats */}
        <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            <div>
              <div className="font-medium text-green-600">
                {devices.filter(d => d.status === 'ACTIVE' || d.status === 'connected').length}
              </div>
              <div className="text-gray-500">Active</div>
            </div>
            <div>
              <div className="font-medium text-yellow-600">
                {devices.filter(d => getBatteryLevel(d) < 20).length}
              </div>
              <div className="text-gray-500">Low Battery</div>
            </div>
            <div>
              <div className="font-medium text-red-600">
                {devices.filter(d => d.status === 'INACTIVE' || d.status === 'disconnected').length}
              </div>
              <div className="text-gray-500">Offline</div>
            </div>
          </div>
        </div>

        {/* Add Device Button */}
        <Button variant="outline" size="sm" className="w-full">
          <Smartphone className="w-4 h-4 mr-2" />
          Add New Device
        </Button>
      </div>
    </Card>
  )
}

export default DeviceStatusGrid