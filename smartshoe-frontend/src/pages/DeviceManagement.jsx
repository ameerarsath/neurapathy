import React, { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { smartShoeAPI } from '../services/api'
import { useAuth } from '../contexts/AuthContext'
import LoadingSpinner from '../components/common/LoadingSpinner'
import DeviceModal from '../components/DeviceModal'
import { 
  Smartphone, 
  Plus, 
  Search, 
  Battery, 
  Wifi, 
  Settings,
  AlertTriangle,
  CheckCircle,
  Clock,
  User,
  Edit,
  Trash2
} from 'lucide-react'
import toast from 'react-hot-toast'

const DeviceManagement = () => {
  const { canAccess } = useAuth()
  const [searchTerm, setSearchTerm] = useState('')
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [showAddModal, setShowAddModal] = useState(false)
  const [editingDevice, setEditingDevice] = useState(null)
  const queryClient = useQueryClient()

  // Debounce search term
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchTerm(searchTerm)
    }, 300)
    return () => clearTimeout(timer)
  }, [searchTerm])

  const { data: devices, isLoading, error } = useQuery({
    queryKey: ['devices', debouncedSearchTerm, statusFilter],
    queryFn: async () => {
      if (statusFilter && statusFilter !== 'all') {
        return smartShoeAPI.devices.getByStatus(statusFilter)
      } else {
        return smartShoeAPI.devices.getAll()
      }
    },
    enabled: canAccess('PROVIDER'),
    select: data => data.data?.devices || data.data || []
  })

  if (!canAccess('PROVIDER')) {
    return (
      <div className="text-center py-12">
        <Smartphone className="h-12 w-12 text-neutral-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-neutral-900 mb-2">Access Restricted</h3>
        <p className="text-neutral-600">You don't have permission to manage devices.</p>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  // Filter devices based on search term
  const displayDevices = Array.isArray(devices) ? devices.filter(device => 
    !debouncedSearchTerm || 
    device.serialNumber.toLowerCase().includes(debouncedSearchTerm.toLowerCase()) ||
    device.model.toLowerCase().includes(debouncedSearchTerm.toLowerCase())
  ) : []

  const getStatusColor = (status) => {
    const colors = {
      'ACTIVE': 'bg-green-100 text-green-800',
      'INACTIVE': 'bg-neutral-100 text-neutral-800',
      'MAINTENANCE': 'bg-yellow-100 text-yellow-800',
      'ERROR': 'bg-red-100 text-red-800',
      'LOW_BATTERY': 'bg-orange-100 text-orange-800'
    }
    return colors[status] || colors['INACTIVE']
  }

  const getStatusIcon = (status) => {
    const icons = {
      'ACTIVE': CheckCircle,
      'INACTIVE': Clock,
      'MAINTENANCE': Settings,
      'ERROR': AlertTriangle,
      'LOW_BATTERY': Battery
    }
    const Icon = icons[status] || Clock
    return <Icon className="h-4 w-4" />
  }

  const getBatteryColor = (level) => {
    if (level >= 80) return 'text-green-600'
    if (level >= 50) return 'text-yellow-600'
    if (level >= 20) return 'text-orange-600'
    return 'text-red-600'
  }

  const isOnline = (lastSync) => {
    const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000)
    return new Date(lastSync) > fiveMinutesAgo
  }

  const requiresCalibration = (calibrationDate) => {
    if (!calibrationDate) return true
    const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)
    return new Date(calibrationDate) < thirtyDaysAgo
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="medical-card">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-neutral-900">Device Management</h1>
            <p className="text-neutral-600 mt-1">
              Monitor and manage smart shoe devices
            </p>
          </div>
          <button 
            onClick={() => setShowAddModal(true)}
            className="flex items-center px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 transition-colors"
          >
            <Plus className="h-4 w-4 mr-2" />
            Register Device
          </button>
        </div>
      </div>

      {/* Device Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="medical-card text-center">
          <div className="text-2xl font-bold text-primary-600">
            {displayDevices.filter(d => d.status === 'ACTIVE').length}
          </div>
          <div className="text-sm text-neutral-600 mt-1">Active Devices</div>
        </div>
        <div className="medical-card text-center">
          <div className="text-2xl font-bold text-secondary-600">
            {displayDevices.filter(d => d.patient).length}
          </div>
          <div className="text-sm text-neutral-600 mt-1">Assigned</div>
        </div>
        <div className="medical-card text-center">
          <div className="text-2xl font-bold text-warning">
            {displayDevices.filter(d => d.batteryLevel < 20).length}
          </div>
          <div className="text-sm text-neutral-600 mt-1">Low Battery</div>
        </div>
        <div className="medical-card text-center">
          <div className="text-2xl font-bold text-error">
            {displayDevices.filter(d => requiresCalibration(d.calibrationDate)).length}
          </div>
          <div className="text-sm text-neutral-600 mt-1">Need Calibration</div>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="medical-card">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
          <div className="flex items-center space-x-4">
            <div className="relative">
              <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-neutral-400" />
              <input
                type="text"
                placeholder="Search devices..."
                className="pl-10 pr-4 py-2 border border-neutral-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <select
              className="px-3 py-2 border border-neutral-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
            >
              <option value="all">All Status</option>
              <option value="ACTIVE">Active</option>
              <option value="INACTIVE">Inactive</option>
              <option value="LOW_BATTERY">Low Battery</option>
              <option value="MAINTENANCE">Maintenance</option>
            </select>
          </div>
          <div className="text-sm text-neutral-600">
            {displayDevices.length} devices total
          </div>
        </div>
      </div>

      {/* Devices Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {displayDevices.length === 0 ? (
          <div className="col-span-full text-center py-12">
            <Smartphone className="h-12 w-12 text-neutral-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-neutral-900 mb-2">No devices found</h3>
            <p className="text-neutral-600">
              {debouncedSearchTerm || statusFilter !== 'all' 
                ? 'Try adjusting your search or filter criteria.' 
                : 'Get started by registering your first device.'
              }
            </p>
          </div>
        ) : (
          displayDevices.map((device) => (
          <div key={device.id} className={`device-card ${device.status.toLowerCase()}`}>
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="font-medium text-neutral-900">{device.model}</h3>
                <p className="text-sm text-neutral-500">{device.serialNumber}</p>
              </div>
              <div className="flex items-center space-x-2">
                {isOnline(device.lastSync) ? (
                  <Wifi className="h-4 w-4 text-success" />
                ) : (
                  <Wifi className="h-4 w-4 text-neutral-400" />
                )}
                <span className={`status-indicator ${getStatusColor(device.status)}`}>
                  {getStatusIcon(device.status)}
                  <span className="ml-1">{device.status.replace('_', ' ')}</span>
                </span>
              </div>
            </div>

            {/* Patient Assignment */}
            <div className="mb-4">
              {device.patient ? (
                <div className="flex items-center text-sm">
                  <User className="h-4 w-4 text-neutral-400 mr-2" />
                  <span className="text-neutral-900">
                    {device.patient.firstName} {device.patient.lastName}
                  </span>
                </div>
              ) : (
                <div className="flex items-center text-sm text-neutral-500">
                  <User className="h-4 w-4 text-neutral-400 mr-2" />
                  <span>Unassigned</span>
                </div>
              )}
            </div>

            {/* Device Metrics */}
            <div className="space-y-3">
              {/* Battery Level */}
              <div className="flex items-center justify-between">
                <div className="flex items-center text-sm">
                  <Battery className={`h-4 w-4 mr-2 ${getBatteryColor(device.batteryLevel)}`} />
                  <span>Battery</span>
                </div>
                <span className={`font-medium ${getBatteryColor(device.batteryLevel)}`}>
                  {device.batteryLevel}%
                </span>
              </div>

              {/* Firmware Version */}
              <div className="flex items-center justify-between text-sm">
                <span className="text-neutral-600">Firmware</span>
                <span className="font-mono text-neutral-900">{device.firmwareVersion}</span>
              </div>

              {/* Calibration Status */}
              <div className="flex items-center justify-between text-sm">
                <span className="text-neutral-600">Calibration</span>
                <span className={`${device.isCalibrated && !requiresCalibration(device.calibrationDate) ? 'text-success' : 'text-warning'}`}>
                  {device.isCalibrated && !requiresCalibration(device.calibrationDate) ? 'Valid' : 'Required'}
                </span>
              </div>

              {/* Last Sync */}
              <div className="flex items-center justify-between text-sm">
                <span className="text-neutral-600">Last Sync</span>
                <span className="text-neutral-900">
                  {new Date(device.lastSync).toLocaleTimeString()}
                </span>
              </div>
            </div>

            {/* Actions */}
            <div className="mt-4 pt-4 border-t border-neutral-200">
              <div className="flex space-x-2">
                <button 
                  onClick={() => setEditingDevice(device)}
                  className="flex-1 px-3 py-2 text-sm bg-primary-50 text-primary-700 rounded hover:bg-primary-100 transition-colors flex items-center justify-center"
                >
                  <Edit className="h-3 w-3 mr-1" />
                  Edit
                </button>
                <button 
                  onClick={() => handleCalibrateDevice(device.id)}
                  className="flex-1 px-3 py-2 text-sm bg-secondary-50 text-secondary-700 rounded hover:bg-secondary-100 transition-colors flex items-center justify-center"
                >
                  <Settings className="h-3 w-3 mr-1" />
                  Calibrate
                </button>
                <button 
                  onClick={() => handleDeleteDevice(device.id)}
                  className="px-3 py-2 text-sm bg-red-50 text-red-700 rounded hover:bg-red-100 transition-colors flex items-center justify-center"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
            </div>
          </div>
          ))
        )}
      </div>

      {/* Add Device Modal */}
      {showAddModal && (
        <DeviceModal 
          isOpen={showAddModal}
          onClose={() => setShowAddModal(false)}
          onSuccess={() => {
            setShowAddModal(false)
            queryClient.invalidateQueries(['devices'])
          }}
        />
      )}

      {/* Edit Device Modal */}
      {editingDevice && (
        <DeviceModal 
          isOpen={!!editingDevice}
          device={editingDevice}
          onClose={() => setEditingDevice(null)}
          onSuccess={() => {
            setEditingDevice(null)
            queryClient.invalidateQueries(['devices'])
          }}
        />
      )}
    </div>
  )

  // Handle device calibration
  async function handleCalibrateDevice(deviceId) {
    try {
      await smartShoeAPI.devices.calibrate(deviceId)
      toast.success('Device calibrated successfully')
      queryClient.invalidateQueries(['devices'])
    } catch (error) {
      toast.error('Failed to calibrate device')
    }
  }

  // Handle device deletion
  async function handleDeleteDevice(deviceId) {
    if (window.confirm('Are you sure you want to deactivate this device?')) {
      try {
        await smartShoeAPI.devices.delete(deviceId)
        toast.success('Device deactivated successfully')
        queryClient.invalidateQueries(['devices'])
      } catch (error) {
        toast.error('Failed to deactivate device')
      }
    }
  }
}

export default DeviceManagement