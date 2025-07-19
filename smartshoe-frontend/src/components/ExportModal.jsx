import React, { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { smartShoeAPI } from '../services/api'
import { useAuth } from '../contexts/AuthContext'
import { X, Download, FileText, File, FileSpreadsheet, Calendar, Shield } from 'lucide-react'
import { handleFileDownload, generateFilename } from '../utils/fileDownload'
import toast from 'react-hot-toast'

const ExportModal = ({ isOpen, onClose, patientId = null }) => {
  const { user, canAccess } = useAuth()
  const [exportFormat, setExportFormat] = useState('csv')
  const [dateRange, setDateRange] = useState({
    startDate: '',
    endDate: ''
  })
  const [useDateRange, setUseDateRange] = useState(false)

  // Determine the actual patient ID to use for export
  const getExportPatientId = () => {
    // If user is a patient, they can only export their own data
    if (user?.role === 'PATIENT') {
      return user.id
    }
    // Admins and providers can export specific patient data or all data
    return patientId
  }

  // Export mutations
  const exportMutation = useMutation({
    mutationFn: async ({ format, patientId, dateRange, useDateRange }) => {
      const exportPatientId = getExportPatientId()
      
      if (useDateRange && dateRange.startDate && dateRange.endDate) {
        // Export with date range - only for admins/providers
        if (!canAccess('PROVIDER')) {
          throw new Error('Date range export not available for patients')
        }
        if (format === 'csv') {
          return smartShoeAPI.medicalReadings.exportDateRangeToCSV(
            dateRange.startDate + 'T00:00:00',
            dateRange.endDate + 'T23:59:59'
          )
        }
        throw new Error('Date range export only available for CSV format')
      } else if (exportPatientId) {
        // Export patient-specific data
        if (format === 'csv') {
          return smartShoeAPI.medicalReadings.exportPatientToCSV(exportPatientId)
        } else if (format === 'pdf') {
          return smartShoeAPI.medicalReadings.exportPatientToPDF(exportPatientId)
        }
        throw new Error('Patient export only available for CSV and PDF formats')
      } else {
        // Export all data - only for admins/providers
        if (!canAccess('PROVIDER')) {
          throw new Error('Full system export not available for patients')
        }
        if (format === 'csv') {
          return smartShoeAPI.medicalReadings.exportToCSV()
        } else if (format === 'excel') {
          return smartShoeAPI.medicalReadings.exportToExcel()
        } else if (format === 'pdf') {
          return smartShoeAPI.medicalReadings.exportToPDF()
        }
      }
    },
    onSuccess: (response) => {
      // Handle file download
      const exportPatientId = getExportPatientId()
      let filename = generateFilename('medical_readings', exportFormat)
      
      if (exportPatientId) {
        const patientLabel = user?.role === 'PATIENT' ? 'my' : `patient_${exportPatientId}`
        filename = generateFilename(`${patientLabel}_readings`, exportFormat)
      }
      
      if (useDateRange) {
        filename = generateFilename(`readings_${dateRange.startDate}_to_${dateRange.endDate}`, exportFormat)
      }
      
      handleFileDownload(response, filename)
      toast.success(`Data exported successfully as ${exportFormat.toUpperCase()}`)
      onClose()
    },
    onError: (error) => {
      console.error('Export error:', error)
      toast.error('Failed to export data. Please try again.')
    }
  })

  const handleExport = () => {
    if (useDateRange && (!dateRange.startDate || !dateRange.endDate)) {
      toast.error('Please select both start and end dates')
      return
    }

    if (useDateRange && new Date(dateRange.startDate) > new Date(dateRange.endDate)) {
      toast.error('Start date must be before end date')
      return
    }

    exportMutation.mutate({
      format: exportFormat,
      patientId,
      dateRange,
      useDateRange
    })
  }

  const getFormatIcon = (format) => {
    switch (format) {
      case 'csv':
        return <FileText className="h-5 w-5" />
      case 'excel':
        return <FileSpreadsheet className="h-5 w-5" />
      case 'pdf':
        return <File className="h-5 w-5" />
      default:
        return <FileText className="h-5 w-5" />
    }
  }

  const getFormatDescription = (format) => {
    switch (format) {
      case 'csv':
        return 'Comma-separated values file, compatible with Excel and other spreadsheet applications'
      case 'excel':
        return 'Excel workbook with formatted data and styling'
      case 'pdf':
        return 'PDF document with formatted tables and summary information'
      default:
        return ''
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-200">
          <h2 className="text-xl font-semibold text-neutral-900">
            Export Medical Readings
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-neutral-100 rounded-full transition-colors"
          >
            <X className="h-5 w-5 text-neutral-500" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Export Format */}
          <div>
            <label className="block text-sm font-medium text-neutral-700 mb-3">
              Export Format
            </label>
            <div className="space-y-3">
              {['csv', 'excel', 'pdf'].map((format) => (
                <label key={format} className="flex items-start space-x-3 cursor-pointer">
                  <input
                    type="radio"
                    name="format"
                    value={format}
                    checked={exportFormat === format}
                    onChange={(e) => setExportFormat(e.target.value)}
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      {getFormatIcon(format)}
                      <span className="font-medium text-neutral-900 uppercase">
                        {format}
                      </span>
                    </div>
                    <p className="text-sm text-neutral-600 mt-1">
                      {getFormatDescription(format)}
                    </p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Date Range Option - Only for Admins/Providers */}
          {!patientId && canAccess('PROVIDER') && (
            <div>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useDateRange}
                  onChange={(e) => setUseDateRange(e.target.checked)}
                  className="rounded border-neutral-300"
                />
                <span className="text-sm font-medium text-neutral-700">
                  Export specific date range
                </span>
              </label>
              
              {useDateRange && (
                <div className="mt-3 space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 mb-1">
                      Start Date
                    </label>
                    <input
                      type="date"
                      value={dateRange.startDate}
                      onChange={(e) => setDateRange(prev => ({ ...prev, startDate: e.target.value }))}
                      className="w-full px-3 py-2 border border-neutral-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 mb-1">
                      End Date
                    </label>
                    <input
                      type="date"
                      value={dateRange.endDate}
                      onChange={(e) => setDateRange(prev => ({ ...prev, endDate: e.target.value }))}
                      className="w-full px-3 py-2 border border-neutral-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Export Info */}
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <Calendar className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium text-blue-900">
                  {user?.role === 'PATIENT' ? 'My Data Export' : 
                   patientId ? 'Patient Export' : 'Full Export'}
                </p>
                <p className="text-sm text-blue-700">
                  {user?.role === 'PATIENT' 
                    ? 'This will export only your personal medical readings.'
                    : patientId 
                      ? 'This will export all medical readings for the selected patient.'
                      : useDateRange 
                        ? 'This will export medical readings within the selected date range.'
                        : 'This will export all medical readings in the system.'
                  }
                </p>
              </div>
            </div>
          </div>

          {/* Patient Security Notice */}
          {user?.role === 'PATIENT' && (
            <div className="bg-green-50 rounded-lg p-4">
              <div className="flex items-start space-x-3">
                <Shield className="h-5 w-5 text-green-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-green-900">
                    Privacy Protected
                  </p>
                  <p className="text-sm text-green-700">
                    You can only export your own medical data. Other patients' information is kept private and secure.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex justify-end space-x-3 p-6 border-t border-neutral-200">
          <button
            onClick={onClose}
            className="px-4 py-2 text-neutral-700 bg-white border border-neutral-300 rounded-md hover:bg-neutral-50 focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            Cancel
          </button>
          <button
            onClick={handleExport}
            disabled={exportMutation.isLoading}
            className="inline-flex items-center px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {exportMutation.isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Exporting...
              </>
            ) : (
              <>
                <Download className="h-4 w-4 mr-2" />
                Export Data
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

export default ExportModal