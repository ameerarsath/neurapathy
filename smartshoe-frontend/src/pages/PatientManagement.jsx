import React, { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { smartShoeAPI } from '../services/api'
import { useAuth } from '../contexts/AuthContext'
import LoadingSpinner from '../components/common/LoadingSpinner'
import { 
  Users, 
  Plus, 
  Search, 
  Filter,
  MoreHorizontal,
  Eye,
  Edit,
  Trash2,
  Calendar,
  X,
  Save
} from 'lucide-react'
import { Link } from 'react-router-dom'
import toast from 'react-hot-toast'
import PatientModal from '../components/PatientModal'

const PatientManagement = () => {
  const { canAccess } = useAuth()
  const [searchTerm, setSearchTerm] = useState('')
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState('')
  const [filterType, setFilterType] = useState('all')
  const [showAddModal, setShowAddModal] = useState(false)
  const [editingPatient, setEditingPatient] = useState(null)
  const queryClient = useQueryClient()

  // Debounce search term
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchTerm(searchTerm)
    }, 300)

    return () => clearTimeout(timer)
  }, [searchTerm])

  const { data: patients, isLoading, error } = useQuery({
    queryKey: ['patients', debouncedSearchTerm, filterType],
    queryFn: async () => {
      if (debouncedSearchTerm) {
        return smartShoeAPI.patients.search(debouncedSearchTerm)
      } else if (filterType && filterType !== 'all') {
        return smartShoeAPI.patients.getByDiabetesType(filterType)
      } else {
        return smartShoeAPI.patients.getAll()
      }
    },
    enabled: canAccess('PROVIDER'),
    select: data => data.data?.patients || data.data || []
  })

  if (!canAccess('PROVIDER')) {
    return (
      <div className="text-center py-12">
        <Users className="h-12 w-12 text-neutral-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-neutral-900 mb-2">Access Restricted</h3>
        <p className="text-neutral-600">You don't have permission to manage patients.</p>
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

  if (error) {
    return (
      <div className="medical-card text-center py-12">
        <h3 className="text-lg font-medium text-error mb-2">Error Loading Patients</h3>
        <p className="text-neutral-600">Unable to fetch patient data. Please try again.</p>
      </div>
    )
  }

  // Removed mock data - using only backend API data

  const displayPatients = Array.isArray(patients) ? patients : []

  const getDiabetesTypeLabel = (type) => {
    const labels = {
      'TYPE_1': 'Type 1',
      'TYPE_2': 'Type 2',
      'GESTATIONAL': 'Gestational',
      'OTHER': 'Other'
    }
    return labels[type] || type
  }

  const getDiabetesTypeColor = (type) => {
    const colors = {
      'TYPE_1': 'bg-red-100 text-red-800',
      'TYPE_2': 'bg-blue-100 text-blue-800',
      'GESTATIONAL': 'bg-purple-100 text-purple-800',
      'OTHER': 'bg-neutral-100 text-neutral-800'
    }
    return colors[type] || colors['OTHER']
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
      {/* Header */}
      <div className="medical-card">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-neutral-900">Patient Management</h1>
            <p className="text-neutral-600 mt-1">
              Manage patient records and medical information
            </p>
          </div>
          <button 
            onClick={() => setShowAddModal(true)}
            className="flex items-center px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 transition-colors"
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Patient
          </button>
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
                placeholder="Search patients..."
                className="pl-10 pr-4 py-2 border border-neutral-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <select
              className="px-3 py-2 border border-neutral-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
            >
              <option value="all">All Types</option>
              <option value="TYPE_1">Type 1</option>
              <option value="TYPE_2">Type 2</option>
              <option value="GESTATIONAL">Gestational</option>
              <option value="OTHER">Other</option>
            </select>
          </div>
          <div className="text-sm text-neutral-600">
            {displayPatients.length} patients total
          </div>
        </div>
      </div>

      {/* Patients Table */}
      <div className="medical-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="medical-table">
            <thead>
              <tr>
                <th>Patient</th>
                <th>Age</th>
                <th>Diabetes Type</th>
                <th>Diagnosis Date</th>
                <th>Last Reading</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-200">
              {displayPatients.length === 0 ? (
                <tr>
                  <td colSpan="7" className="text-center py-12">
                    <Users className="h-12 w-12 text-neutral-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-neutral-900 mb-2">No patients found</h3>
                    <p className="text-neutral-600">
                      {debouncedSearchTerm || filterType !== 'all' 
                        ? 'Try adjusting your search or filter criteria.' 
                        : 'Get started by adding your first patient.'
                      }
                    </p>
                  </td>
                </tr>
              ) : (
                displayPatients.map((patient) => (
                <tr key={patient.id} className="hover:bg-neutral-50">
                  <td>
                    <div>
                      <p className="font-medium text-neutral-900">
                        {patient.firstName} {patient.lastName}
                      </p>
                      <p className="text-sm text-neutral-500">{patient.email}</p>
                    </div>
                  </td>
                  <td className="text-neutral-900">
                    {calculateAge(patient.dateOfBirth)}
                  </td>
                  <td>
                    <span className={`status-indicator ${getDiabetesTypeColor(patient.diabetesType)}`}>
                      {getDiabetesTypeLabel(patient.diabetesType)}
                    </span>
                  </td>
                  <td className="text-neutral-900">
                    {patient.diagnosisDate ? new Date(patient.diagnosisDate).toLocaleDateString() : 'N/A'}
                  </td>
                  <td className="text-neutral-900">
                    {patient.lastReading ? new Date(patient.lastReading).toLocaleDateString() : 'No readings yet'}
                  </td>
                  <td>
                    <span className={`status-indicator ${patient.isActive ? 'status-normal' : 'status-inactive'}`}>
                      {patient.isActive ? 'Active' : 'Inactive'}
                    </span>
                  </td>
                  <td>
                    <div className="flex items-center space-x-2">
                      <Link
                        to={`/patients/${patient.id}`}
                        className="p-1 text-neutral-400 hover:text-primary-600"
                      >
                        <Eye className="h-4 w-4" />
                      </Link>
                      <button 
                        onClick={() => setEditingPatient(patient)}
                        className="p-1 text-neutral-400 hover:text-secondary-600"
                      >
                        <Edit className="h-4 w-4" />
                      </button>
                      <button 
                        onClick={() => handleDeletePatient(patient.id)}
                        className="p-1 text-neutral-400 hover:text-error"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add Patient Modal */}
      {showAddModal && (
        <PatientModal 
          isOpen={showAddModal}
          onClose={() => setShowAddModal(false)}
          onSuccess={() => {
            setShowAddModal(false)
            queryClient.invalidateQueries(['patients'])
          }}
        />
      )}

      {/* Edit Patient Modal */}
      {editingPatient && (
        <PatientModal 
          isOpen={!!editingPatient}
          patient={editingPatient}
          onClose={() => setEditingPatient(null)}
          onSuccess={() => {
            setEditingPatient(null)
            queryClient.invalidateQueries(['patients'])
          }}
        />
      )}
    </div>
  )

  // Handle patient deletion
  async function handleDeletePatient(patientId) {
    if (window.confirm('Are you sure you want to deactivate this patient?')) {
      try {
        await smartShoeAPI.patients.delete(patientId)
        toast.success('Patient deactivated successfully')
        queryClient.invalidateQueries(['patients'])
      } catch (error) {
        toast.error('Failed to deactivate patient')
      }
    }
  }
}

export default PatientManagement