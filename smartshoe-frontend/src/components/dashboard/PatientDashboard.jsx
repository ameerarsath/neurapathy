import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { smartShoeAPI } from '../../services/api'
import StatsCard from './StatsCard'
import {
  Heart,
  Battery,
  TrendingUp,
  Activity,
  AlertTriangle,
  Target,
  Clock,
  Footprints
} from 'lucide-react'

const PatientDashboard = ({ user }) => {
  // Fetch patient-specific dashboard data
  const { data: dashboardData, isLoading: loadingDashboard } = useQuery({
    queryKey: ['dashboard-data', user?.id],
    queryFn: () => smartShoeAPI.dashboard.getData(user?.id),
    enabled: !!user?.id,
    select: data => data?.data || data
  })

  // Fetch patient's recent readings
  const { data: recentReadings } = useQuery({
    queryKey: ['patient-readings', user?.id],
    queryFn: () => smartShoeAPI.medicalReadings.getByPatient(user?.id),
    enabled: !!user?.id,
    select: data => {
      // Handle the API response structure: {data: {readings: [...], total: N}}
      const readings = data?.data?.readings || data?.data || []
      return Array.isArray(readings) ? readings.slice(0, 5) : []
    }
  })

  // Mock patient-specific data (would come from API in real implementation)
  const patientData = {
    dailySteps: dashboardData?.dailySteps || 8247,
    stepsGoal: dashboardData?.stepsGoal || 10000,
    stepsTrend: dashboardData?.stepsTrend || +12.3,
    pressureStatus: dashboardData?.pressureStatus || "Normal",
    batteryLevel: dashboardData?.batteryLevel || 78,
    lastTestDate: "2024-01-15",
    nextAppointment: "2024-01-22",
    totalTests: 24,
    weeklyAvgSteps: 7823
  }

  const stepsProgress = (patientData.dailySteps / patientData.stepsGoal) * 100

  return (
    <div className="space-y-6">
      {/* Patient Header */}
      <div className="medical-card">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-neutral-900 flex items-center">
              <Heart className="h-6 w-6 text-success mr-2" />
              My Health Dashboard
            </h1>
            <p className="text-neutral-600 mt-1">
              Personal health monitoring and insights
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm text-success">
              <div className="h-2 w-2 bg-success rounded-full animate-pulse"></div>
              <span>Device Connected</span>
            </div>
          </div>
        </div>
      </div>

      {/* Patient Stats Grid */}
      <div className="dashboard-grid">
        <StatsCard
          title="Today's Steps"
          value={patientData.dailySteps.toLocaleString()}
          icon={Footprints}
          color="primary"
          trend={patientData.stepsTrend}
          subtitle={`Goal: ${patientData.stepsGoal.toLocaleString()}`}
          isLoading={loadingDashboard}
        />
        <StatsCard
          title="Pressure Status"
          value={patientData.pressureStatus}
          icon={Heart}
          color={patientData.pressureStatus === "Normal" ? "success" : "warning"}
          subtitle="Foot pressure analysis"
          isLoading={loadingDashboard}
        />
        <StatsCard
          title="Device Battery"
          value={`${patientData.batteryLevel}%`}
          icon={Battery}
          color={patientData.batteryLevel > 50 ? "success" : patientData.batteryLevel > 20 ? "warning" : "error"}
          subtitle={patientData.batteryLevel > 20 ? "Good level" : "Needs charging"}
          isLoading={loadingDashboard}
        />
        <StatsCard
          title="Weekly Average"
          value={patientData.weeklyAvgSteps.toLocaleString()}
          icon={TrendingUp}
          color="info"
          trend={+8.5}
          subtitle="Steps per day"
          isLoading={loadingDashboard}
        />
      </div>

      {/* Activity Progress */}
      <div className="medical-card">
        <h3 className="text-lg font-semibold text-neutral-900 mb-4">Daily Activity Progress</h3>
        <div className="space-y-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-neutral-700">Steps Goal</span>
              <span className="text-sm text-neutral-600">
                {patientData.dailySteps.toLocaleString()} / {patientData.stepsGoal.toLocaleString()}
              </span>
            </div>
            <div className="w-full bg-neutral-200 rounded-full h-3">
              <div 
                className="bg-primary-500 h-3 rounded-full transition-all duration-300"
                style={{ width: `${Math.min(stepsProgress, 100)}%` }}
              ></div>
            </div>
            <p className="text-xs text-neutral-600 mt-1">
              {stepsProgress >= 100 ? "Goal achieved! 🎉" : `${(100 - stepsProgress).toFixed(0)}% to go`}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Test Results */}
        <div className="medical-card">
          <h3 className="text-lg font-semibold text-neutral-900 mb-4">Recent Test Results</h3>
          <div className="space-y-3">
            {recentReadings?.length > 0 ? (
              recentReadings.map((reading, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-neutral-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <Activity className="h-5 w-5 text-primary-600" />
                    <div>
                      <p className="font-medium text-neutral-900">Neuropathy Test</p>
                      <p className="text-sm text-neutral-600">
                        {new Date(reading.timestamp).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      reading.severity === 'NORMAL' ? 'bg-green-100 text-green-800' :
                      reading.severity === 'MILD' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {reading.severity || 'Normal'}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-6">
                <Activity className="h-12 w-12 text-neutral-300 mx-auto mb-3" />
                <p className="text-neutral-600">No recent test results</p>
                <p className="text-sm text-neutral-500">Schedule a test to see your results here</p>
              </div>
            )}
          </div>
        </div>

        {/* Upcoming & Important Info */}
        <div className="medical-card">
          <h3 className="text-lg font-semibold text-neutral-900 mb-4">Important Information</h3>
          <div className="space-y-4">
            <div className="flex items-center space-x-3 p-3 bg-blue-50 rounded-lg">
              <Clock className="h-5 w-5 text-blue-600" />
              <div>
                <p className="font-medium text-neutral-900">Next Appointment</p>
                <p className="text-sm text-neutral-600">{patientData.nextAppointment}</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
              <Target className="h-5 w-5 text-green-600" />
              <div>
                <p className="font-medium text-neutral-900">Total Tests Completed</p>
                <p className="text-sm text-neutral-600">{patientData.totalTests} tests</p>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-3 bg-purple-50 rounded-lg">
              <Heart className="h-5 w-5 text-purple-600" />
              <div>
                <p className="font-medium text-neutral-900">Last Test</p>
                <p className="text-sm text-neutral-600">{patientData.lastTestDate}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions for Patients */}
      <div className="medical-card">
        <h3 className="text-lg font-semibold text-neutral-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="p-4 bg-primary-50 hover:bg-primary-100 rounded-lg text-left transition-colors">
            <Activity className="h-6 w-6 text-primary-600 mb-2" />
            <p className="font-medium text-neutral-900">Take New Test</p>
            <p className="text-sm text-neutral-600">Start neuropathy assessment</p>
          </button>
          
          <button className="p-4 bg-secondary-50 hover:bg-secondary-100 rounded-lg text-left transition-colors">
            <TrendingUp className="h-6 w-6 text-secondary-600 mb-2" />
            <p className="font-medium text-neutral-900">View Progress</p>
            <p className="text-sm text-neutral-600">See your health trends</p>
          </button>
          
          <button className="p-4 bg-success hover:bg-green-100 rounded-lg text-left transition-colors">
            <Heart className="h-6 w-6 text-green-600 mb-2" />
            <p className="font-medium text-neutral-900">Health Tips</p>
            <p className="text-sm text-neutral-600">Personalized recommendations</p>
          </button>
        </div>
      </div>
    </div>
  )
}

export default PatientDashboard