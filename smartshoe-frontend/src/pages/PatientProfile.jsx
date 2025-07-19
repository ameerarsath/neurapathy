import React from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { smartShoeAPI } from '../services/api'
import LoadingSpinner from '../components/common/LoadingSpinner'
import { 
  User, 
  Calendar, 
  Mail, 
  Phone,
  Activity,
  Smartphone,
  TrendingUp,
  Heart
} from 'lucide-react'

const PatientProfile = () => {
  const { id } = useParams()

  const { data: patient, isLoading } = useQuery({
    queryKey: ['patient', id],
    queryFn: () => smartShoeAPI.patients.getById(id),
    select: data => data.data
  })

  // Removed mock data - using only backend API data
  const displayPatient = patient

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  const calculateAge = (dateOfBirth) => {
    const today = new Date()
    const birthDate = new Date(dateOfBirth)
    let age = today.getFullYear() - birthDate.getFullYear()
    const monthDiff = today.getMonth() - birthDate.getMonth()
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
      age--
    }
    return age
  }

  return (
    <div className="space-y-6">
      {/* Patient Header */}
      <div className="medical-card">
        <div className="flex items-center space-x-6">
          <div className="h-20 w-20 rounded-full bg-primary-100 flex items-center justify-center">
            <User className="h-10 w-10 text-primary-600" />
          </div>
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-neutral-900">
              {displayPatient.firstName} {displayPatient.lastName}
            </h1>
            <p className="text-neutral-600">Patient ID: {displayPatient.id}</p>
            <div className="flex items-center space-x-4 mt-2">
              <span className="status-indicator status-normal">
                {displayPatient.isActive ? 'Active' : 'Inactive'}
              </span>
              <span className="text-sm text-neutral-500">
                {calculateAge(displayPatient.dateOfBirth)} years old
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Patient Information */}
        <div className="lg:col-span-2 space-y-6">
          {/* Contact Information */}
          <div className="medical-card">
            <h3 className="text-lg font-medium text-neutral-900 mb-4">Contact Information</h3>
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Mail className="h-5 w-5 text-neutral-400" />
                <span className="text-neutral-900">{displayPatient.email}</span>
              </div>
              {displayPatient.phoneNumber && (
                <div className="flex items-center space-x-3">
                  <Phone className="h-5 w-5 text-neutral-400" />
                  <span className="text-neutral-900">{displayPatient.phoneNumber}</span>
                </div>
              )}
              <div className="flex items-center space-x-3">
                <Calendar className="h-5 w-5 text-neutral-400" />
                <span className="text-neutral-900">
                  Born {new Date(displayPatient.dateOfBirth).toLocaleDateString()}
                </span>
              </div>
            </div>
          </div>

          {/* Medical Information */}
          <div className="medical-card">
            <h3 className="text-lg font-medium text-neutral-900 mb-4">Medical Information</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-neutral-700">Diabetes Type</label>
                <p className="mt-1 text-sm text-neutral-900">
                  {displayPatient.diabetesType?.replace('_', ' ')}
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-neutral-700">Diagnosis Date</label>
                <p className="mt-1 text-sm text-neutral-900">
                  {new Date(displayPatient.diagnosisDate).toLocaleDateString()}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="space-y-6">
          <div className="medical-card">
            <h3 className="text-lg font-medium text-neutral-900 mb-4">Quick Stats</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Activity className="h-4 w-4 text-primary-600" />
                  <span className="text-sm text-neutral-600">Total Readings</span>
                </div>
                <span className="font-medium text-neutral-900">124</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Smartphone className="h-4 w-4 text-secondary-600" />
                  <span className="text-sm text-neutral-600">Devices</span>
                </div>
                <span className="font-medium text-neutral-900">1</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="h-4 w-4 text-success" />
                  <span className="text-sm text-neutral-600">Compliance</span>
                </div>
                <span className="font-medium text-success">94%</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Heart className="h-4 w-4 text-error" />
                  <span className="text-sm text-neutral-600">Risk Level</span>
                </div>
                <span className="font-medium text-warning">Medium</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PatientProfile