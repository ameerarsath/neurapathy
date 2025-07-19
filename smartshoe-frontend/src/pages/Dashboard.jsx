import React from 'react'
import { useAuth } from '../contexts/AuthContext'
import LoadingSpinner from '../components/common/LoadingSpinner'
import AdminDashboard from '../components/dashboard/AdminDashboard'
import DoctorDashboard from '../components/dashboard/DoctorDashboard'
import PatientDashboard from '../components/dashboard/PatientDashboard'

const Dashboard = () => {
  const { user, isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  // Render role-specific dashboard
  switch (user?.role) {
    case 'ADMIN':
      return <AdminDashboard user={user} />
    case 'PROVIDER':
      return <DoctorDashboard user={user} />
    case 'PATIENT':
      return <PatientDashboard user={user} />
    default:
      return <PatientDashboard user={user} />
  }
}

export default Dashboard