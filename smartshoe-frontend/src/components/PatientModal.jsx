import React, { useState, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { smartShoeAPI } from '../services/api'
import { X, Save, User, Mail, Phone, Calendar, Heart } from 'lucide-react'
import toast from 'react-hot-toast'

const PatientModal = ({ isOpen, patient, onClose, onSuccess }) => {
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    dateOfBirth: '',
    phoneNumber: '',
    diabetesType: '',
    diagnosisDate: ''
  })

  const [errors, setErrors] = useState({})

  // Populate form data when editing
  useEffect(() => {
    if (patient) {
      setFormData({
        firstName: patient.firstName || '',
        lastName: patient.lastName || '',
        email: patient.email || '',
        dateOfBirth: patient.dateOfBirth || '',
        phoneNumber: patient.phoneNumber || '',
        diabetesType: patient.diabetesType || '',
        diagnosisDate: patient.diagnosisDate || ''
      })
    } else {
      setFormData({
        firstName: '',
        lastName: '',
        email: '',
        dateOfBirth: '',
        phoneNumber: '',
        diabetesType: '',
        diagnosisDate: ''
      })
    }
    setErrors({})
  }, [patient])

  // Create patient mutation
  const createPatientMutation = useMutation({
    mutationFn: (data) => smartShoeAPI.patients.create(data),
    onSuccess: (response) => {
      toast.success('Patient created successfully!')
      onSuccess()
    },
    onError: (error) => {
      const errorMessage = error.response?.data?.message || 'Failed to create patient'
      toast.error(errorMessage)
      if (error.response?.data?.errors) {
        setErrors(error.response.data.errors)
      }
    }
  })

  // Update patient mutation
  const updatePatientMutation = useMutation({
    mutationFn: ({ id, data }) => smartShoeAPI.patients.update(id, data),
    onSuccess: (response) => {
      toast.success('Patient updated successfully!')
      onSuccess()
    },
    onError: (error) => {
      const errorMessage = error.response?.data?.message || 'Failed to update patient'
      toast.error(errorMessage)
      if (error.response?.data?.errors) {
        setErrors(error.response.data.errors)
      }
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

    if (!formData.firstName.trim()) {
      newErrors.firstName = 'First name is required'
    }
    if (!formData.lastName.trim()) {
      newErrors.lastName = 'Last name is required'
    }
    if (!formData.email.trim()) {
      newErrors.email = 'Email is required'
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email must be valid'
    }
    if (!formData.dateOfBirth) {
      newErrors.dateOfBirth = 'Date of birth is required'
    } else {
      const birthDate = new Date(formData.dateOfBirth)
      const today = new Date()
      if (birthDate >= today) {
        newErrors.dateOfBirth = 'Date of birth must be in the past'
      }
    }
    if (formData.phoneNumber && !/^[+]?[1-9][0-9-]{0,14}$/.test(formData.phoneNumber)) {
      newErrors.phoneNumber = 'Phone number must be valid'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!validateForm()) {
      return
    }

    if (patient) {
      // Update existing patient
      updatePatientMutation.mutate({ 
        id: patient.id, 
        data: formData 
      })
    } else {
      // Create new patient
      createPatientMutation.mutate(formData)
    }
  }

  const isLoading = createPatientMutation.isLoading || updatePatientMutation.isLoading

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-200">
          <h2 className="text-xl font-semibold text-neutral-900">
            {patient ? 'Edit Patient' : 'Add New Patient'}
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
          {/* Personal Information */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-neutral-900 flex items-center">
              <User className="h-5 w-5 mr-2" />
              Personal Information
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-1">
                  First Name *
                </label>
                <input
                  type="text"
                  name="firstName"
                  value={formData.firstName}
                  onChange={handleInputChange}
                  className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                    errors.firstName ? 'border-red-500' : 'border-neutral-300'
                  }`}
                  placeholder="Enter first name"
                />
                {errors.firstName && (
                  <p className="text-red-500 text-sm mt-1">{errors.firstName}</p>
                )}
              </div>
              
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-1">
                  Last Name *
                </label>
                <input
                  type="text"
                  name="lastName"
                  value={formData.lastName}
                  onChange={handleInputChange}
                  className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                    errors.lastName ? 'border-red-500' : 'border-neutral-300'
                  }`}
                  placeholder="Enter last name"
                />
                {errors.lastName && (
                  <p className="text-red-500 text-sm mt-1">{errors.lastName}</p>
                )}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-neutral-700 mb-1 flex items-center">
                <Mail className="h-4 w-4 mr-1" />
                Email Address *
              </label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                  errors.email ? 'border-red-500' : 'border-neutral-300'
                }`}
                placeholder="Enter email address"
              />
              {errors.email && (
                <p className="text-red-500 text-sm mt-1">{errors.email}</p>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-1 flex items-center">
                  <Calendar className="h-4 w-4 mr-1" />
                  Date of Birth *
                </label>
                <input
                  type="date"
                  name="dateOfBirth"
                  value={formData.dateOfBirth}
                  onChange={handleInputChange}
                  className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                    errors.dateOfBirth ? 'border-red-500' : 'border-neutral-300'
                  }`}
                />
                {errors.dateOfBirth && (
                  <p className="text-red-500 text-sm mt-1">{errors.dateOfBirth}</p>
                )}
              </div>
              
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-1 flex items-center">
                  <Phone className="h-4 w-4 mr-1" />
                  Phone Number
                </label>
                <input
                  type="tel"
                  name="phoneNumber"
                  value={formData.phoneNumber}
                  onChange={handleInputChange}
                  className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                    errors.phoneNumber ? 'border-red-500' : 'border-neutral-300'
                  }`}
                  placeholder="Enter phone number"
                />
                {errors.phoneNumber && (
                  <p className="text-red-500 text-sm mt-1">{errors.phoneNumber}</p>
                )}
              </div>
            </div>
          </div>

          {/* Medical Information */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-neutral-900 flex items-center">
              <Heart className="h-5 w-5 mr-2" />
              Medical Information
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-1">
                  Diabetes Type
                </label>
                <select
                  name="diabetesType"
                  value={formData.diabetesType}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-neutral-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="">Select diabetes type</option>
                  <option value="TYPE_1">Type 1</option>
                  <option value="TYPE_2">Type 2</option>
                  <option value="GESTATIONAL">Gestational</option>
                  <option value="OTHER">Other</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-1">
                  Diagnosis Date
                </label>
                <input
                  type="date"
                  name="diagnosisDate"
                  value={formData.diagnosisDate}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-neutral-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
            </div>
          </div>

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
                  {patient ? 'Updating...' : 'Creating...'}
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  {patient ? 'Update Patient' : 'Create Patient'}
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default PatientModal