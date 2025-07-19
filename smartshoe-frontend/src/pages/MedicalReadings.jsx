import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { smartShoeAPI } from '../services/api'
import { useAuth } from '../contexts/AuthContext'
import LoadingSpinner from '../components/common/LoadingSpinner'
import ExportModal from '../components/ExportModal'
import { 
  Activity, 
  Search, 
  Filter,
  Calendar,
  TrendingUp,
  AlertTriangle,
  Eye,
  Download
} from 'lucide-react'

const MedicalReadings = () => {
  const { user, canAccess } = useAuth()
  const [searchTerm, setSearchTerm] = useState('')
  const [typeFilter, setTypeFilter] = useState('all')
  const [severityFilter, setSeverityFilter] = useState('all')
  const [showExportModal, setShowExportModal] = useState(false)

  const { data: readings, isLoading } = useQuery({
    queryKey: ['medical-readings', typeFilter, severityFilter, user?.id],
    queryFn: () => {
      // If user is a patient and has a valid patient ID, fetch their readings
      if (user?.role === 'PATIENT' && user?.id) {
        return smartShoeAPI.medicalReadings.getByPatient(user.id)
      }
      // Admins and providers can see all readings
      return smartShoeAPI.medicalReadings.getAll()
    },
    enabled: canAccess('PATIENT'),
    select: data => {
      // Handle different response structures
      if (data?.data?.readings) {
        return data.data.readings  // Patient-specific endpoint structure
      }
      return data?.data || data    // All readings endpoint structure
    }
  })

  // Removed mock data - using only backend API data
  const displayReadings = Array.isArray(readings) ? readings : []

  const getSeverityColor = (severity) => {
    const colors = {
      'NORMAL': 'bg-green-100 text-green-800',
      'MILD': 'bg-yellow-100 text-yellow-800',
      'MODERATE': 'bg-orange-100 text-orange-800',
      'SEVERE': 'bg-red-100 text-red-800',
      'CRITICAL': 'bg-red-200 text-red-900'
    }
    return colors[severity] || colors['NORMAL']
  }

  const getReadingTypeIcon = (type) => {
    const icons = {
      'PRESSURE': '💪',
      'VIBRATION': '📳',
      'TEMPERATURE': '🌡️',
      'PAIN_ASSESSMENT': '😣',
      'BLOOD_GLUCOSE': '🩸',
      'FOOT_SCAN': '👣',
      'NEUROPATHY_SCREENING': '🧠'
    }
    return icons[type] || '📊'
  }

  const getQualityColor = (score) => {
    if (score >= 90) return 'text-green-600'
    if (score >= 80) return 'text-yellow-600'
    if (score >= 70) return 'text-orange-600'
    return 'text-red-600'
  }

  if (!canAccess('PATIENT')) {
    return (
      <div className="text-center py-12">
        <Activity className="h-12 w-12 text-neutral-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-neutral-900 mb-2">Access Restricted</h3>
        <p className="text-neutral-600">You don't have permission to view medical readings.</p>
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="medical-card">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-neutral-900 flex items-center">
              <Activity className="h-6 w-6 text-primary-600 mr-2" />
              {user?.role === 'PATIENT' ? 'My Medical Readings' : 'Medical Readings'}
            </h1>
            <p className="text-neutral-600 mt-1">
              {user?.role === 'PATIENT' 
                ? 'Your personal sensor data and medical measurements' 
                : 'Sensor data and medical measurements from smart shoes'
              }
            </p>
          </div>
          <button 
            onClick={() => setShowExportModal(true)}
            className="flex items-center px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 transition-colors"
          >
            <Download className="h-4 w-4 mr-2" />
            Export Data
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="medical-card text-center">
          <div className="text-2xl font-bold text-primary-600">
            {displayReadings.length}
          </div>
          <div className="text-sm text-neutral-600 mt-1">Total Readings</div>
        </div>
        <div className="medical-card text-center">
          <div className="text-2xl font-bold text-success">
            {displayReadings.filter(r => r.severityLevel === 'NORMAL').length}
          </div>
          <div className="text-sm text-neutral-600 mt-1">Normal</div>
        </div>
        <div className="medical-card text-center">
          <div className="text-2xl font-bold text-warning">
            {displayReadings.filter(r => ['MILD', 'MODERATE'].includes(r.severityLevel)).length}
          </div>
          <div className="text-sm text-neutral-600 mt-1">Attention Needed</div>
        </div>
        <div className="medical-card text-center">
          <div className="text-2xl font-bold text-error">
            {displayReadings.filter(r => ['SEVERE', 'CRITICAL'].includes(r.severityLevel)).length}
          </div>
          <div className="text-sm text-neutral-600 mt-1">Critical</div>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="medical-card">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
          <div className="flex flex-col sm:flex-row sm:items-center space-y-4 sm:space-y-0 sm:space-x-4">
            <div className="relative">
              <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-neutral-400" />
              <input
                type="text"
                placeholder="Search readings..."
                className="pl-10 pr-4 py-2 border border-neutral-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <select
              className="px-3 py-2 border border-neutral-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
            >
              <option value="all">All Types</option>
              <option value="PRESSURE">Pressure</option>
              <option value="VIBRATION">Vibration</option>
              <option value="TEMPERATURE">Temperature</option>
              <option value="PAIN_ASSESSMENT">Pain Assessment</option>
            </select>
            <select
              className="px-3 py-2 border border-neutral-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
            >
              <option value="all">All Severity</option>
              <option value="NORMAL">Normal</option>
              <option value="MILD">Mild</option>
              <option value="MODERATE">Moderate</option>
              <option value="SEVERE">Severe</option>
              <option value="CRITICAL">Critical</option>
            </select>
          </div>
          <div className="flex items-center space-x-2 text-sm text-neutral-600">
            <Calendar className="h-4 w-4" />
            <span>Last 30 days</span>
          </div>
        </div>
      </div>

      {/* Readings List */}
      <div className="space-y-4">
        {displayReadings.map((reading) => (
          <div key={reading.id} className="medical-card hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-4">
                {/* Reading Type Icon */}
                <div className="text-2xl">
                  {getReadingTypeIcon(reading.readingType)}
                </div>
                
                {/* Reading Details */}
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <h3 className="font-medium text-neutral-900">
                      {reading.readingType.replace('_', ' ')} Reading
                    </h3>
                    <span className={`status-indicator ${getSeverityColor(reading.severityLevel)}`}>
                      {reading.severityLevel}
                    </span>
                    {reading.isBaseline && (
                      <span className="status-indicator bg-blue-100 text-blue-800">
                        Baseline
                      </span>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-neutral-600">Patient: </span>
                      <span className="font-medium">
                        {reading.patient.firstName} {reading.patient.lastName}
                      </span>
                    </div>
                    <div>
                      <span className="text-neutral-600">Value: </span>
                      <span className="font-medium">
                        {reading.value} {reading.unit}
                      </span>
                    </div>
                    <div>
                      <span className="text-neutral-600">Foot: </span>
                      <span className="font-medium">{reading.footSide}</span>
                    </div>
                    <div>
                      <span className="text-neutral-600">Device: </span>
                      <span className="font-medium">{reading.device.serialNumber}</span>
                    </div>
                    <div>
                      <span className="text-neutral-600">Quality: </span>
                      <span className={`font-medium ${getQualityColor(reading.qualityScore)}`}>
                        {reading.qualityScore}%
                      </span>
                    </div>
                    <div>
                      <span className="text-neutral-600">Recorded: </span>
                      <span className="font-medium">
                        {new Date(reading.recordedAt).toLocaleString()}
                      </span>
                    </div>
                  </div>
                  
                  {reading.notes && (
                    <div className="mt-3 p-3 bg-neutral-50 rounded-md">
                      <p className="text-sm text-neutral-700">{reading.notes}</p>
                    </div>
                  )}
                  
                  {reading.hasMotionArtifacts && (
                    <div className="mt-2 flex items-center text-xs text-warning">
                      <AlertTriangle className="h-3 w-3 mr-1" />
                      Motion artifacts detected
                    </div>
                  )}
                </div>
              </div>
              
              {/* Actions */}
              <div className="flex items-center space-x-2">
                <button className="p-2 text-neutral-400 hover:text-primary-600 rounded-md hover:bg-primary-50">
                  <Eye className="h-4 w-4" />
                </button>
                <button className="p-2 text-neutral-400 hover:text-secondary-600 rounded-md hover:bg-secondary-50">
                  <TrendingUp className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {/* Export Modal */}
      {showExportModal && (
        <ExportModal 
          isOpen={showExportModal}
          onClose={() => setShowExportModal(false)}
        />
      )}
    </div>
  )
}

export default MedicalReadings