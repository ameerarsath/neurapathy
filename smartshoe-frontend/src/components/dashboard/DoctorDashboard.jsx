import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { smartShoeAPI } from '../../services/api'
import StatsCard from './StatsCard'
import QuickActions from './QuickActions'
import RecentActivity from './RecentActivity'
import {
  Users,
  Smartphone,
  Activity,
  AlertTriangle,
  Stethoscope,
  Brain,
  Heart
} from 'lucide-react'

const DoctorDashboard = ({ user }) => {
  // Fetch statistics relevant to doctors/providers
  const { data: patientStats, isLoading: loadingPatients } = useQuery({
    queryKey: ['patient-statistics'],
    queryFn: () => smartShoeAPI.patients.getStatistics(),
    select: data => data?.data?.statistics || data?.statistics || data
  })

  const { data: deviceStats, isLoading: loadingDevices } = useQuery({
    queryKey: ['device-statistics'],
    queryFn: () => smartShoeAPI.devices.getStatistics(),
    select: data => data?.data?.statistics || data?.statistics || data
  })

  const { data: readingStats, isLoading: loadingReadings } = useQuery({
    queryKey: ['reading-statistics'],
    queryFn: () => smartShoeAPI.medicalReadings.getStatistics(),
    select: data => data?.data?.statistics || data?.statistics || data
  })

  const { data: criticalReadings } = useQuery({
    queryKey: ['critical-readings'],
    queryFn: () => smartShoeAPI.medicalReadings.getCritical(),
    select: data => data?.data || data
  })

  const { data: recentPatients } = useQuery({
    queryKey: ['recent-patients'],
    queryFn: () => smartShoeAPI.patients.getAll(),
    select: data => {
      // Handle the API response structure: {data: {patients: [...]} or {data: [...]}}
      const patients = data?.data?.patients || data?.data || []
      return Array.isArray(patients) ? patients.slice(0, 5) : []
    }
  })

  const isLoading = loadingPatients || loadingDevices || loadingReadings

  // Calculate some derived stats
  const normalReadings = readingStats?.normalReadings || 0
  const abnormalReadings = readingStats?.abnormalReadings || 0
  const totalReadings = readingStats?.totalReadings || 0
  const abnormalPercentage = totalReadings > 0 ? ((abnormalReadings / totalReadings) * 100).toFixed(1) : 0

  return (
    <div className="space-y-6">
      {/* Doctor Header */}
      <div className="medical-card">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-neutral-900 flex items-center">
              <Stethoscope className="h-6 w-6 text-secondary-600 mr-2" />
              Provider Dashboard
            </h1>
            <p className="text-neutral-600 mt-1">
              Patient care and medical data management
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm text-success">
              <div className="h-2 w-2 bg-success rounded-full animate-pulse"></div>
              <span>Connected to {deviceStats?.activeDevices || 0} devices</span>
            </div>
          </div>
        </div>
      </div>

      {/* Provider Stats Grid */}
      <div className="dashboard-grid">
        <StatsCard
          title="My Patients"
          value={patientStats?.totalActivePatients || 0}
          icon={Users}
          color="primary"
          trend={+5.2}
          subtitle="Under your care"
          isLoading={isLoading}
        />
        <StatsCard
          title="Active Devices"
          value={deviceStats?.activeDevices || 0}
          icon={Smartphone}
          color="secondary"
          trend={+2.1}
          subtitle="Currently monitoring"
          isLoading={isLoading}
        />
        <StatsCard
          title="Recent Readings"
          value={readingStats?.totalReadings || 0}
          icon={Activity}
          color="success"
          trend={+12.5}
          subtitle="Last 24 hours"
          isLoading={isLoading}
        />
        <StatsCard
          title="Alerts Pending"
          value={criticalReadings?.length || 0}
          icon={AlertTriangle}
          color="error"
          trend={-3.2}
          subtitle="Need review"
          isLoading={isLoading}
        />
      </div>

      {/* Medical Insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatsCard
          title="Normal Readings"
          value={normalReadings}
          icon={Heart}
          color="success"
          trend={+8.1}
          subtitle="Within normal range"
          isLoading={isLoading}
        />
        <StatsCard
          title="Abnormal Readings"
          value={abnormalReadings}
          icon={Brain}
          color="warning"
          trend={-4.2}
          subtitle={`${abnormalPercentage}% of total`}
          isLoading={isLoading}
        />
        <StatsCard
          title="Critical Cases"
          value={readingStats?.criticalReadings || 0}
          icon={AlertTriangle}
          color="error"
          trend={-1.5}
          subtitle="Immediate attention"
          isLoading={isLoading}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Quick Actions */}
        <div className="lg:col-span-1">
          <QuickActions />
        </div>

        {/* Recent Activity */}
        <div className="lg:col-span-2">
          <RecentActivity />
        </div>
      </div>

      {/* Recent Patients */}
      <div className="medical-card">
        <h3 className="text-lg font-semibold text-neutral-900 mb-4">Recent Patients</h3>
        <div className="space-y-3">
          {recentPatients?.map((patient) => (
            <div key={patient.id} className="flex items-center justify-between p-3 bg-neutral-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="h-10 w-10 bg-primary-100 rounded-full flex items-center justify-center">
                  <span className="text-primary-600 font-medium text-sm">
                    {patient.firstName?.charAt(0)}{patient.lastName?.charAt(0)}
                  </span>
                </div>
                <div>
                  <p className="font-medium text-neutral-900">
                    {patient.firstName} {patient.lastName}
                  </p>
                  <p className="text-sm text-neutral-600">
                    {patient.diabetesType} • Age {patient.age}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm text-neutral-600">Last reading</p>
                <p className="text-xs text-neutral-500">2 hours ago</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default DoctorDashboard