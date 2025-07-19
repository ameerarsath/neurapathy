import React, { useState, useEffect } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { smartShoeAPI } from '../services/api'
import { X, Save, Smartphone, Battery, Wifi, Settings, User } from 'lucide-react'
import toast from 'react-hot-toast'

const DeviceModal = ({ isOpen, device, onClose, onSuccess }) => {
  const [formData, setFormData] = useState({
    serialNumber: '',
    model: '',
    firmwareVersion: '',
    deviceType: 'SMART_SHOE',
    batteryLevel: 100,
    patientId: ''
  })

  const [errors, setErrors] = useState({})

  // Get patients for assignment
  const { data: patients } = useQuery({
    queryKey: ['patients'],
    queryFn: () => smartShoeAPI.patients.getAll(),
    select: data => data.data?.patients || data.data || []
  })

  // Populate form data when editing
  useEffect(() => {
    if (device) {
      setFormData({
        serialNumber: device.serialNumber || '',
        model: device.model || '',
        firmwareVersion: device.firmwareVersion || '',
        deviceType: device.deviceType || 'SMART_SHOE',
        batteryLevel: device.batteryLevel || 100,
        patientId: device.patient?.id || ''
      })
    } else {
      setFormData({
        serialNumber: '',
        model: '',
        firmwareVersion: '',
        deviceType: 'SMART_SHOE',
        batteryLevel: 100,
        patientId: ''
      })
    }
    setErrors({})
  }, [device])

  // Register device mutation
  const registerDeviceMutation = useMutation({
    mutationFn: (data) => smartShoeAPI.devices.create(data),
    onSuccess: (response) => {
      toast.success('Device registered successfully!')
      onSuccess()
    },
    onError: (error) => {
      const errorMessage = error.response?.data?.message || 'Failed to register device'
      toast.error(errorMessage)
      if (error.response?.data?.errors) {
        setErrors(error.response.data.errors)
      }
    }
  })

  // Update device mutation
  const updateDeviceMutation = useMutation({
    mutationFn: ({ id, data }) => smartShoeAPI.devices.update(id, data),
    onSuccess: (response) => {
      toast.success('Device updated successfully!')
      onSuccess()
    },
    onError: (error) => {
      const errorMessage = error.response?.data?.message || 'Failed to update device'
      toast.error(errorMessage)
      if (error.response?.data?.errors) {
        setErrors(error.response.data.errors)
      }
    }
  })

  // Assign device mutation
  const assignDeviceMutation = useMutation({
    mutationFn: ({ deviceId, patientId }) => smartShoeAPI.devices.assignToPatient(deviceId, patientId),
    onSuccess: (response) => {
      toast.success('Device assigned successfully!')
      onSuccess()
    },
    onError: (error) => {
      const errorMessage = error.response?.data?.message || 'Failed to assign device'
      toast.error(errorMessage)
    }
  })

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }))
    }
  }

  const validateForm = () => {
    const newErrors = {}

    if (!formData.serialNumber.trim()) {
      newErrors.serialNumber = 'Serial number is required'
    } else if (formData.serialNumber.length > 100) {
      newErrors.serialNumber = 'Serial number must not exceed 100 characters'
    }

    if (!formData.model.trim()) {
      newErrors.model = 'Device model is required'
    } else if (formData.model.length > 50) {
      newErrors.model = 'Model must not exceed 50 characters'
    }

    if (!formData.firmwareVersion.trim()) {
      newErrors.firmwareVersion = 'Firmware version is required'
    }

    if (formData.batteryLevel < 0 || formData.batteryLevel > 100) {
      newErrors.batteryLevel = 'Battery level must be between 0 and 100'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!validateForm()) {
      return
    }

    // Prepare device data
    const deviceData = {
      serialNumber: formData.serialNumber,
      model: formData.model,
      firmwareVersion: formData.firmwareVersion,
      deviceType: formData.deviceType,
      batteryLevel: parseInt(formData.batteryLevel)
    }

    if (device) {
      // Update existing device
      updateDeviceMutation.mutate({ 
        id: device.id, 
        data: deviceData 
      })
    } else {
      // Register new device
      registerDeviceMutation.mutate(deviceData)
    }
  }

  const handlePatientAssignment = async () => {
    if (!formData.patientId || !device) return

    assignDeviceMutation.mutate({
      deviceId: device.id,
      patientId: parseInt(formData.patientId)
    })
  }

  const isLoading = registerDeviceMutation.isLoading || updateDeviceMutation.isLoading || assignDeviceMutation.isLoading

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-200">
          <h2 className="text-xl font-semibold text-neutral-900">
            {device ? 'Edit Device' : 'Register New Device'}
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-neutral-100 rounded-full transition-colors"
          >
            <X className="h-5 w-5 text-neutral-500" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Device Information */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-neutral-900 flex items-center">
              <Smartphone className="h-5 w-5 mr-2" />
              Device Information
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-1">
                  Serial Number *
                </label>
                <input
                  type="text"
                  name="serialNumber"
                  value={formData.serialNumber}
                  onChange={handleInputChange}
                  className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                    errors.serialNumber ? 'border-red-500' : 'border-neutral-300'
                  }`}
                  placeholder="Enter serial number"
                />
                {errors.serialNumber && (
                  <p className="text-red-500 text-sm mt-1">{errors.serialNumber}</p>
                )}
              </div>
              
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-1">
                  Device Model *
                </label>
                <input
                  type="text"
                  name="model"
                  value={formData.model}
                  onChange={handleInputChange}
                  className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                    errors.model ? 'border-red-500' : 'border-neutral-300'
                  }`}
                  placeholder="Enter device model"
                />
                {errors.model && (
                  <p className="text-red-500 text-sm mt-1">{errors.model}</p>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-1">
                  Firmware Version *
                </label>
                <input
                  type="text"
                  name="firmwareVersion"
                  value={formData.firmwareVersion}
                  onChange={handleInputChange}
                  className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                    errors.firmwareVersion ? 'border-red-500' : 'border-neutral-300'
                  }`}
                  placeholder="Enter firmware version"
                />
                {errors.firmwareVersion && (
                  <p className="text-red-500 text-sm mt-1">{errors.firmwareVersion}</p>
                )}
              </div>
              
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-1">
                  Device Type
                </label>
                <select
                  name="deviceType"
                  value={formData.deviceType}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-neutral-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="SMART_SHOE">Smart Shoe</option>
                  <option value="SENSOR_INSOLE">Sensor Insole</option>
                  <option value="TEMPERATURE_SENSOR">Temperature Sensor</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-neutral-700 mb-1 flex items-center">
                <Battery className="h-4 w-4 mr-1" />
                Battery Level (%)
              </label>
              <input
                type="number"
                name="batteryLevel"
                value={formData.batteryLevel}
                onChange={handleInputChange}
                min="0"
                max="100"
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                  errors.batteryLevel ? 'border-red-500' : 'border-neutral-300'
                }`}
              />
              {errors.batteryLevel && (
                <p className="text-red-500 text-sm mt-1">{errors.batteryLevel}</p>
              )}
            </div>
          </div>

          {/* Patient Assignment (only for existing devices) */}
          {device && (
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-neutral-900 flex items-center">
                <User className="h-5 w-5 mr-2" />
                Patient Assignment
              </h3>
              
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-1">
                  Assign to Patient
                </label>
                <div className="flex space-x-2">
                  <select
                    name="patientId"
                    value={formData.patientId}
                    onChange={handleInputChange}
                    className="flex-1 px-3 py-2 border border-neutral-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                  >
                    <option value="">Select patient</option>
                    {patients?.map(patient => (
                      <option key={patient.id} value={patient.id}>
                        {patient.firstName} {patient.lastName}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={handlePatientAssignment}
                    disabled={!formData.patientId || isLoading}
                    className="px-4 py-2 bg-secondary-600 text-white rounded-md hover:bg-secondary-700 focus:outline-none focus:ring-2 focus:ring-secondary-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Assign
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end space-x-3 pt-6 border-t border-neutral-200">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-neutral-700 bg-white border border-neutral-300 rounded-md hover:bg-neutral-50 focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="inline-flex items-center px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  {device ? 'Updating...' : 'Registering...'}
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  {device ? 'Update Device' : 'Register Device'}
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default DeviceModal