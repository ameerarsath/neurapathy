import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { smartShoeAPI } from '../../services/api'
import StatsCard from './StatsCard'
import QuickActions from './QuickActions'
import RecentActivity from './RecentActivity'
import SystemStatus from './SystemStatus'
import {
  Users,
  Smartphone,
  Activity,
  AlertTriangle,
  Shield,
  UserCheck,
  Settings
} from 'lucide-react'

const AdminDashboard = ({ user }) => {
  // Fetch all statistics for admin view
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

  const { data: userStats, isLoading: loadingUsers } = useQuery({
    queryKey: ['user-statistics'],
    queryFn: () => smartShoeAPI.dashboard.getStatistics(),
    select: data => data?.data?.statistics || data?.statistics || data
  })

  const { data: criticalReadings } = useQuery({
    queryKey: ['critical-readings'],
    queryFn: () => smartShoeAPI.medicalReadings.getCritical(),
    select: data => data?.data || data
  })

  const { data: lowBatteryDevices } = useQuery({
    queryKey: ['low-battery-devices'],
    queryFn: () => smartShoeAPI.devices.getLowBattery(),
    select: data => data?.data || data
  })

  const isLoading = loadingPatients || loadingDevices || loadingReadings || loadingUsers

  return (
    <div className="space-y-6">
      {/* Admin Header */}
      <div className="medical-card">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-neutral-900 flex items-center">
              <Shield className="h-6 w-6 text-primary-600 mr-2" />
              Admin Dashboard
            </h1>
            <p className="text-neutral-600 mt-1">
              Complete system administration and oversight
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm text-success">
              <div className="h-2 w-2 bg-success rounded-full animate-pulse"></div>
              <span>System Online</span>
            </div>
          </div>
        </div>
      </div>

      {/* Admin Stats Grid */}
      <div className="dashboard-grid">
        <StatsCard
          title="Total Patients"
          value={patientStats?.totalActivePatients || 0}
          icon={Users}
          color="primary"
          trend={+5.2}
          subtitle="Active patients"
          isLoading={isLoading}
        />
        <StatsCard
          title="Total Devices"
          value={deviceStats?.totalActiveDevices || 0}
          icon={Smartphone}
          color="secondary"
          trend={+2.1}
          subtitle="All devices"
          isLoading={isLoading}
        />
        <StatsCard
          title="System Users"
          value={userStats?.totalUsers || 0}
          icon={UserCheck}
          color="success"
          trend={+8.3}
          subtitle="Platform users"
          isLoading={isLoading}
        />
        <StatsCard
          title="Critical Alerts"
          value={criticalReadings?.length || 0}
          icon={AlertTriangle}
          color="error"
          trend={-3.2}
          subtitle="Require attention"
          isLoading={isLoading}
        />
      </div>

      {/* Additional Admin Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatsCard
          title="Medical Readings"
          value={readingStats?.totalReadings || 0}
          icon={Activity}
          color="info"
          trend={+12.5}
          subtitle="Total recorded"
          isLoading={isLoading}
        />
        <StatsCard
          title="Active Devices"
          value={deviceStats?.activeDevices || 0}
          icon={Smartphone}
          color="success"
          trend={+4.1}
          subtitle="Currently active"
          isLoading={isLoading}
        />
        <StatsCard
          title="Low Battery"
          value={lowBatteryDevices?.length || 0}
          icon={AlertTriangle}
          color="warning"
          trend={-2.3}
          subtitle="Need charging"
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

      {/* System Status */}
      <SystemStatus 
        criticalAlerts={criticalReadings?.length || 0}
        lowBatteryDevices={lowBatteryDevices?.length || 0}
      />
    </div>
  )
}

export default AdminDashboard